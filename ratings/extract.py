"""
extract.py — Stage 4: Structured LLM Extraction.

Retrieves stored chunks for a given scheme from ChromaDB and uses an OpenAI
LLM (via ``instructor``) to extract structured ``CertificationCriteria``
Pydantic objects from each chunk.  Results are written as a JSON array to
``data/processed/{scheme}/criteria.json``.

``instructor`` wraps the OpenAI client so that the model is forced to respond
with a JSON payload that matches the Pydantic schema — retrying automatically
on validation failure.

CLI usage
---------
python -m ratings.extract --scheme BREEAM
python -m ratings.extract --scheme BREEAM --limit 20
"""

from __future__ import annotations

import json
from typing import Optional

import instructor
import typer
from openai import OpenAI

from ratings.config import (
    CHUNKS_DIR,
    LLM_MODEL,
    OPENAI_API_KEY,
    PROCESSED_DIR,
)
from ratings.schemas import CertificationCriteria

app = typer.Typer(
    name="extract",
    help="Stage 4 — Extract structured criteria from chunks using an LLM.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert in building sustainability certification systems (BREEAM, LEED, DGNB, BNB).

Your task is to extract a SINGLE certification criterion from the supplied text chunk.
Populate every field of the schema as accurately as possible using all context provided.

Guidelines:
- Use the "Section context" (ancestor headings) to identify the credit name and category
  even when the chunk itself shows only a sub-section like "Beschreibung" or "Bewertung".
- `credit_name` must be the name of the criterion (e.g. "Treibhauspotenzial (GWP)",
  "BN 1.1.1", "Mat 01").  Do NOT use section labels like "Beschreibung" as the name.
  If you truly cannot identify a criterion name, set it to "Unknown".
- `category` should reflect the top-level chapter or criterion group (e.g. "Ökologische
  Qualität", "Materials & Resources").
- `source_document` should be the human-readable document title, not the filename.
- `page_number` should be extracted from ``<!-- Page N -->`` markers visible in the
  chunk or its section context.

For the boolean flags:
- `touches_material_reduction`: set True if the criterion incentivises reducing
  material quantities, embodied carbon, or improving material efficiency.
- `touches_structural_efficiency`: set True if the criterion incentivises structural
  efficiency, load-bearing optimisation, or reducing structural material consumption.
"""

_USER_TEMPLATE = """\
Certification scheme: {scheme}
Source document: {source_document}
{section_context_block}
--- BEGIN CHUNK ---
{text}
--- END CHUNK ---

Extract the certification criterion described above.
"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def get_chunks_for_scheme(scheme: str, limit: Optional[int] = None) -> list[dict]:
    """Load chunks from the JSONL file produced by Stage 3 (ratings.chunk).

    Parameters
    ----------
    scheme:
        Certification scheme identifier (e.g. ``"BNB"``).
    limit:
        Maximum number of chunks to return.  ``None`` returns all.

    Returns
    -------
    list[dict]
        Each dict contains ``"text"``, ``"metadata"``, and ``"id"`` keys.

    Raises
    ------
    FileNotFoundError
        If the JSONL chunks file does not exist yet.
    ValueError
        If the JSONL file contains no chunks.
    """
    jsonl_path = CHUNKS_DIR / f"{scheme}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {jsonl_path}\n"
            f"Run Stage 3 first:  python -m ratings.chunk --scheme {scheme}"
        )

    chunks: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            section_context = " > ".join(record.get("heading_path", []))
            page_range = record.get("page_range")
            page_number = page_range if isinstance(page_range, int) else None
            chunk_id = f"{record.get('source_document', 'unknown')}::{record.get('heading', '')}"
            chunks.append({
                "id": chunk_id,
                "text": record.get("body", ""),
                "metadata": {
                    # Pass every field from the JSONL record so that nothing
                    # defined in the markdown frontmatter is silently dropped.
                    "source_document": record.get("source_document", "unknown"),
                    "document_title": record.get("source_document", "unknown"),
                    "section_context": section_context,
                    "page_number": page_number,
                    "scheme": record.get("scheme", scheme),
                    "version": record.get("version"),
                    "content_type": record.get("content_type"),
                    "credit_id": record.get("credit_id"),
                    "credit_name": record.get("credit_name"),
                    "category": record.get("category"),
                    "building_type": record.get("building_type"),
                    "role": record.get("role"),
                    "heading": record.get("heading"),
                    "extra_frontmatter": record.get("extra_frontmatter", {}),
                },
            })

    if not chunks:
        raise ValueError(
            f"No chunks found in {jsonl_path}.  "
            f"Run Stage 3 first:  python -m ratings.chunk --scheme {scheme}"
        )

    if limit is not None:
        chunks = chunks[:limit]

    return chunks


def extract_criterion(
    client: instructor.Instructor,
    chunk: dict,
    scheme: str,
) -> Optional[CertificationCriteria]:
    """Call the LLM to extract a ``CertificationCriteria`` from *chunk*.

    Parameters
    ----------
    client:
        Instructor-patched OpenAI client.
    chunk:
        Dict with ``"text"`` and ``"metadata"`` keys.
    scheme:
        Certification scheme identifier — passed as context to the prompt.

    Returns
    -------
    CertificationCriteria | None
        Extracted criterion, or ``None`` if the LLM call fails after retries.
    """
    meta = chunk["metadata"] or {}
    source_document = meta.get("document_title") or meta.get("source_document", "unknown")
    section_context = meta.get("section_context", "")
    page_from_meta: Optional[int] = meta.get("page_number")

    section_context_block = (
        f"Section context (ancestor headings): {section_context}\n"
        if section_context
        else ""
    )

    user_message = _USER_TEMPLATE.format(
        scheme=scheme,
        source_document=source_document,
        section_context_block=section_context_block,
        text=chunk["text"],
    )

    try:
        criterion: CertificationCriteria = client.chat.completions.create(
            model=LLM_MODEL,
            response_model=CertificationCriteria,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_retries=3,
        )
        # Patch fields from chunk metadata.
        # Rule: any value explicitly defined in the markdown frontmatter is
        # authoritative and ALWAYS overrides what the LLM produced.  The LLM
        # only fills in fields absent from the frontmatter (e.g. description,
        # benchmark_value, the boolean flags).
        patch: dict = {}

        # Fields with a 1-to-1 mapping between frontmatter key and model field
        _FM_OVERRIDES: list[tuple[str, str]] = [
            ("credit_id",      "credit_id"),
            ("credit_name",    "credit_name"),
            ("category",       "category"),
            ("building_type",  "building_type"),
            ("scheme",         "scheme_name"),
            ("source_document","source_document"),
            ("version",        "version"),
            ("content_type",   "content_type"),
        ]
        for meta_key, model_key in _FM_OVERRIDES:
            val = meta.get(meta_key)
            if val:  # only override when the frontmatter actually had this field
                patch[model_key] = str(val)

        # page_number: only fill in from metadata when the LLM left it null
        # (LLM may extract a more precise in-text page reference)
        if page_from_meta is not None and criterion.page_number is None:
            patch["page_number"] = page_from_meta

        # extra_metadata: serialise any additional frontmatter as a JSON string
        extra_fm = meta.get("extra_frontmatter")
        if extra_fm:
            patch["extra_metadata"] = json.dumps(extra_fm, ensure_ascii=False)

        if patch:
            criterion = criterion.model_copy(update=patch)
        return criterion
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"  Warning: extraction failed for chunk {chunk['id']}: {exc}", err=True)
        return None


def extract_criteria_for_scheme(
    scheme: str,
    limit: Optional[int] = None,
    *,
    verbose: bool = True,
    checkpoint_every: int = 10,
    resume: bool = False,
) -> list[CertificationCriteria]:
    """Extract structured criteria for all chunks belonging to *scheme*.

    Parameters
    ----------
    scheme:
        Certification scheme identifier.
    limit:
        Cap the number of chunks processed.  Useful for testing.
    verbose:
        If *True*, print per-chunk progress.
    checkpoint_every:
        Write an incremental criteria.json every N successful extractions so
        that progress is not lost if the run is interrupted.
    resume:
        If *True* and a criteria.json already exists, skip chunk IDs that
        have already been processed (matched on ``source_document`` count is
        not reliable, so we skip by comparing the saved chunk count).

    Returns
    -------
    list[CertificationCriteria]
        Successfully extracted criteria.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set.  Add it to your .env file or environment."
        )

    chunks = get_chunks_for_scheme(scheme, limit=limit)

    # Resume: skip already-processed chunks by loading previous results
    already_done: list[CertificationCriteria] = []
    skip_count = 0
    if resume:
        try:
            already_done = load_criteria(scheme)
            skip_count = len(already_done)
            if verbose and skip_count:
                typer.echo(f"Resuming: skipping first {skip_count} chunk(s) already processed")
            chunks = chunks[skip_count:]
        except FileNotFoundError:
            pass  # no previous run — start fresh

    if verbose:
        typer.echo(f"Processing {len(chunks)} chunk(s) for scheme: {scheme}")

    raw_client = OpenAI(api_key=OPENAI_API_KEY)
    patched_client = instructor.from_openai(raw_client)

    criteria: list[CertificationCriteria] = list(already_done)
    total_all = len(criteria) + len(chunks)
    for i, chunk in enumerate(chunks, start=skip_count + 1):
        if verbose:
            typer.echo(f"  [{i}/{total_all}] Extracting ...", nl=False)

        criterion = extract_criterion(patched_client, chunk, scheme)

        if criterion is not None:
            criteria.append(criterion)
            if verbose:
                typer.echo(f" -> {criterion.credit_name}")
        else:
            if verbose:
                typer.echo(" -> skipped (error)")

        # Incremental checkpoint
        if len(criteria) % checkpoint_every == 0:
            save_criteria(criteria, scheme)

    return criteria


def save_criteria(criteria: list[CertificationCriteria], scheme: str) -> None:
    """Serialise *criteria* to ``data/processed/{scheme}/criteria.json``.

    Parameters
    ----------
    criteria:
        Validated ``CertificationCriteria`` instances.
    scheme:
        Used to determine the output path.
    """
    out_dir = PROCESSED_DIR / scheme
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "criteria.json"
    payload = [c.model_dump() for c in criteria]
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_criteria(scheme: str) -> list[CertificationCriteria]:
    """Load previously saved criteria JSON for *scheme*.

    Parameters
    ----------
    scheme:
        Certification scheme identifier.

    Returns
    -------
    list[CertificationCriteria]
        Validated Pydantic instances.

    Raises
    ------
    FileNotFoundError
        If the criteria JSON file does not exist yet.
    """
    criteria_path = PROCESSED_DIR / scheme / "criteria.json"
    if not criteria_path.exists():
        raise FileNotFoundError(
            f"Criteria file not found: {criteria_path}\n"
            f"Run Stage 4 first:  python -m ratings.extract --scheme {scheme}"
        )

    raw: list[dict] = json.loads(criteria_path.read_text(encoding="utf-8"))
    return [CertificationCriteria(**item) for item in raw]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    scheme: str = typer.Option(
        ...,
        "--scheme",
        help="Certification scheme identifier, e.g. BREEAM, LEED, DGNB, BNB.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Maximum number of chunks to process.  Omit to process all.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="Skip chunks already in criteria.json and append new results.",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print progress."),
) -> None:
    """Extract structured criteria from stored chunks using an LLM (Stage 4)."""
    scheme = scheme.upper()

    try:
        criteria = extract_criteria_for_scheme(
            scheme, limit=limit, verbose=verbose, resume=resume
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    save_criteria(criteria, scheme)

    if verbose:
        out_path = PROCESSED_DIR / scheme / "criteria.json"
        typer.echo(f"\nExtraction complete: {len(criteria)} criteria saved -> {out_path}")


if __name__ == "__main__":
    app()
