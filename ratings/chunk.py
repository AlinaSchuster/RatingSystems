"""
chunk.py — Stage 3 (chunking) and Stage 4 (interactive review).

Stage 3 — Semantic Chunking
    Reads human-validated markdown files from ``data/cleaned/<scheme>/``,
    splits them at ``[credit]`` and ``[criterion]`` heading boundaries, and
    serialises the resulting ``Chunk`` objects as JSONL to
    ``data/chunks/<scheme>.jsonl``.

    Each chunk captures the full heading hierarchy (``heading_path``),
    structured block-level content (tables, figures, formulas, notes), and
    page range extracted from ``<!-- Page N -->`` markers.

    CLI usage::

        python -m ratings.chunk --scheme BNB

Stage 4 — Interactive Review
    Loads a ``.jsonl`` file produced by Stage 3 and presents each chunk one
    at a time.  The reviewer can accept, edit (opens ``$EDITOR``), skip, or
    quit.  Progress is persisted so the session can be resumed.

    CLI usage::

        python -m ratings.chunk --review BNB

Output locations
    ``data/chunks/<scheme>.jsonl``          — all chunks from Stage 3
    ``data/chunks/reviewed/<scheme>.jsonl`` — accepted/edited chunks
    ``data/chunks/skipped/<scheme>.jsonl``  — skipped chunks
    ``data/chunks/<scheme>.progress.json``  — review progress cursor
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import typer
import yaml

from ratings.config import CHUNKS_DIR, CLEANED_DIR
from ratings.clean import validate_file

log = logging.getLogger(__name__)

app = typer.Typer(
    name="chunk",
    help="Stage 3 — Chunk cleaned markdown; Stage 4 — Review chunks interactively.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ROLE_TAGS: frozenset[str] = frozenset(
    ["category", "credit", "criterion", "prerequisite", "general"]
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_ROLE_TAG_RE = re.compile(r"\[([a-zA-Z]+)\]")
_BLOCK_TAG_RE = re.compile(r"^>\s*\[([a-z]+)\]\s*(.*)")
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_PAGE_MARKER_RE = re.compile(r"<!--\s*Page\s+(\d+)\s*-->")
_NOTE_PAGE_RE = re.compile(r"^>\s*\[note\]\s*Page\s+(\d+)\s*$", re.IGNORECASE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)

# Roles that trigger a new chunk regardless of body content
_CHUNK_ROLES: frozenset[str] = frozenset(["credit", "criterion"])


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single semantic unit extracted from a cleaned markdown file."""

    scheme: str
    version: str
    content_type: str
    heading_path: list[str]           # ancestor headings (outermost first), NOT including self
    heading: str                       # this chunk's own heading text (with role tag)
    role: str                          # role tag string, e.g. "credit"
    body: str                          # raw markdown body
    tables: list[dict] = field(default_factory=list)    # [{"caption": str, "markdown": str}]
    figures: list[dict] = field(default_factory=list)   # [{"caption": str, "extractable": bool}]
    formulas: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source_document: str = ""
    page_range: Optional[int] = None                      # page number at chunk start, or None
    credit_id: str = ""               # scheme's own ID, e.g. "BNB_BN 1.1.1"
    credit_name: str = ""             # full criterion/credit name
    category: str = ""               # top-level group
    building_type: str = ""           # building type, e.g. "Büro- und Verwaltungsgebäude"
    # subcategory removed — hierarchy is encoded in heading_path
    extra_frontmatter: dict = field(default_factory=dict)  # any extra frontmatter keys not mapped above


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_frontmatter_chunk(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_text).

    Raises ``ValueError`` if no frontmatter block is found.
    """
    stripped = text.lstrip("\n\r ")
    m = _FRONTMATTER_RE.match(stripped)
    if m is None:
        raise ValueError("No YAML frontmatter block found (expected --- ... ---)")
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse frontmatter YAML: {exc}") from exc
    body = stripped[m.end():]
    return fm, body


def _extract_role(heading_text: str) -> str:
    """Return the role tag value from a heading, or empty string if absent."""
    for m in _ROLE_TAG_RE.finditer(heading_text):
        tag = m.group(1).lower()
        if tag in VALID_ROLE_TAGS:
            return tag
    return ""


def _extract_page_numbers(text: str) -> list[int]:
    """Return all page numbers from ``> [note] Page N`` or legacy ``<!-- Page N -->`` markers."""
    pages = [int(m) for m in _NOTE_PAGE_RE.findall(text)]
    if not pages:
        pages = [int(m) for m in _PAGE_MARKER_RE.findall(text)]
    return pages


def _strip_page_notes(text: str) -> str:
    """Remove all ``> [note] Page N`` lines from *text*."""
    return "\n".join(
        line for line in text.splitlines()
        if not _NOTE_PAGE_RE.match(line.strip())
    )


def _extract_tagged_blocks(
    body: str,
) -> tuple[list[dict], list[dict], list[str], list[str]]:
    """Parse ``> [tag] ...`` lines to extract tables, figures, formulas, notes.

    Returns
    -------
    tuple
        ``(tables, figures, formulas, notes)``
    """
    tables: list[dict] = []
    figures: list[dict] = []
    formulas: list[str] = []
    notes: list[str] = []

    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        bm = _BLOCK_TAG_RE.match(line.strip())
        if bm:
            tag = bm.group(1).lower()
            caption_or_content = bm.group(2).strip()
            if tag == "table":
                table_lines = []
                j = i + 1
                while j < len(lines) and _TABLE_ROW_RE.match(lines[j]):
                    table_lines.append(lines[j])
                    j += 1
                tables.append(
                    {"caption": caption_or_content, "markdown": "\n".join(table_lines)}
                )
                i = j
                continue
            elif tag == "figure":
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                content_line = lines[j].strip() if j < len(lines) else ""
                extractable = "not extractable" not in (caption_or_content + content_line).lower()
                figures.append({"caption": caption_or_content, "extractable": extractable})
                i = j + 1 if j < len(lines) else i + 1
                continue
            elif tag == "formula":
                formulas.append(caption_or_content)
            elif tag == "note":
                notes.append(caption_or_content)
        i += 1

    return tables, figures, formulas, notes


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------


def parse_chunks(md_path: Path) -> list[Chunk]:
    """Parse a cleaned markdown file into a list of ``Chunk`` objects.

    Sections are created at every heading boundary.  A section is emitted as
    a ``Chunk`` when:

    * its heading has a ``[credit]`` or ``[criterion]`` role, **or**
    * it has accumulated non-whitespace body lines (so no content is lost).

    ``heading_path`` contains the ancestor headings (outermost first) at the
    time the chunk's heading was encountered, i.e. the heading stack *before*
    the current heading was pushed.

    Parameters
    ----------
    md_path:
        Path to a validated cleaned markdown file.

    Returns
    -------
    list[Chunk]
        All extracted chunks in document order.
    """
    text = md_path.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter_chunk(text)

    scheme = str(fm.get("scheme", ""))
    version = str(fm.get("version", ""))
    content_type = str(fm.get("content_type", "general"))
    source_document = md_path.stem
    credit_id = str(fm.get("credit_id", "") or "")
    credit_name = str(fm.get("credit_name", "") or "")
    category = str(fm.get("category", "") or "")
    building_type = str(fm.get("building_type", "") or "")
    # Capture any other frontmatter keys not mapped to dedicated Chunk fields
    _KNOWN_FM_KEYS = {
        "scheme", "version", "content_type", "document",
        "credit_id", "credit_name", "category", "building_type",
    }
    extra_frontmatter = {k: v for k, v in fm.items() if k not in _KNOWN_FM_KEYS}

    # Heading stack: list of (level, heading_text_with_tag)
    heading_stack: list[tuple[int, str]] = []

    # Current section accumulator
    current_heading: Optional[str] = None
    current_role: str = ""
    body_lines: list[str] = []

    # Running page tracker — updated whenever a > [note] Page N is seen
    current_page: Optional[int] = None
    # Page at the start of the current chunk section
    chunk_start_page: Optional[int] = None

    chunks: list[Chunk] = []

    def _flush() -> None:
        """Emit the accumulated section as a Chunk if appropriate."""
        nonlocal current_page, chunk_start_page
        if current_heading is None:
            return
        body_text = "\n".join(body_lines).strip()
        # Update current_page from any page notes in this section's body
        page_nums = _extract_page_numbers(body_text)
        if page_nums:
            current_page = max(page_nums)
        # Strip page note lines from the body
        body_text = _strip_page_notes(body_text).strip()
        if current_role in _CHUNK_ROLES or body_text:
            tbls, figs, fmls, nts = _extract_tagged_blocks(body_text)
            chunks.append(
                Chunk(
                    scheme=scheme,
                    version=version,
                    content_type=content_type,
                    heading_path=[h for _, h in heading_stack[:-1]],
                    heading=current_heading,
                    role=current_role,
                    body=body_text,
                    tables=tbls,
                    figures=figs,
                    formulas=fmls,
                    notes=nts,
                    source_document=source_document,
                    page_range=chunk_start_page,
                    credit_id=credit_id,
                    credit_name=credit_name,
                    category=category,
                    building_type=building_type,
                    extra_frontmatter=extra_frontmatter,
                )
            )

    for line in body.splitlines():
        # Track page notes even between headings so current_page stays up to date
        pm = _NOTE_PAGE_RE.match(line.strip())
        if pm:
            current_page = int(pm.group(1))

        hm = _HEADING_RE.match(line)
        if hm:
            _flush()
            body_lines = []
            # Snapshot the current page as the start page for the new section
            chunk_start_page = current_page

            level = len(hm.group(1))
            heading_text = hm.group(2).strip()
            role = _extract_role(heading_text)

            # Pop stack entries at the same level or deeper
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            # heading_path = current stack (ancestors before we push self)
            current_heading = heading_text
            current_role = role

            heading_stack.append((level, heading_text))
        else:
            if current_heading is not None:
                body_lines.append(line)

    _flush()

    return chunks


def chunk_scheme(scheme: str) -> list[Chunk]:
    """Parse all cleaned markdown files for *scheme* and return all chunks.

    Reads from ``data/cleaned/<scheme>/``.

    Raises
    ------
    FileNotFoundError
        If no cleaned files exist for the scheme.
    """
    cleaned_dir = CLEANED_DIR / scheme.upper()
    if not cleaned_dir.exists():
        raise FileNotFoundError(
            f"Cleaned directory not found: {cleaned_dir}\n"
            f"Run Stage 2 first:  python -m ratings.clean --copy --scheme {scheme}"
        )

    md_files = sorted(cleaned_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No cleaned Markdown files found in {cleaned_dir}.")

    all_chunks: list[Chunk] = []
    blocked: list[str] = []
    for md_path in md_files:
        violations = validate_file(md_path)
        if violations:
            blocked.append(md_path.name)
            typer.echo(
                f"\n  !! STOP: {md_path.name} — {len(violations)} validation error(s)."
                f"\n     Edit the file then run:"
                f"\n       python -m ratings.clean --validate {md_path}"
                f"\n     This file will be skipped until it passes validation.\n"
            )
            continue
        try:
            file_chunks = parse_chunks(md_path)
            all_chunks.extend(file_chunks)
            log.info("  %s -> %d chunk(s)", md_path.name, len(file_chunks))
        except ValueError as exc:
            log.warning("  Skipping %s: %s", md_path.name, exc)

    if blocked:
        typer.echo(
            f"  {len(blocked)} file(s) blocked pending manual edit + validation."
            f" Run Stage 3 again once they pass --validate."
        )

    return all_chunks


def save_chunks(chunks: list[Chunk], out_path: Path) -> None:
    """Serialise *chunks* to JSONL at *out_path* (one JSON object per line)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(dataclasses.asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(scheme: str) -> list[Chunk]:
    """Load chunks from ``data/chunks/<scheme>.jsonl``.

    Raises
    ------
    FileNotFoundError
        If the JSONL file does not exist.
    """
    jsonl_path = CHUNKS_DIR / f"{scheme.upper()}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {jsonl_path}\n"
            f"Run Stage 3 first:  python -m ratings.chunk --scheme {scheme}"
        )
    chunks: list[Chunk] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            chunks.append(Chunk(**json.loads(line)))
    return chunks


# ---------------------------------------------------------------------------
# Review helpers
# ---------------------------------------------------------------------------

_REVIEWED_DIR = "reviewed"
_SKIPPED_DIR = "skipped"


def _progress_path(scheme: str) -> Path:
    return CHUNKS_DIR / f"{scheme.upper()}.progress.json"


def _load_progress(scheme: str) -> int:
    """Return the next chunk index to review (0 if no progress saved)."""
    p = _progress_path(scheme)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")).get("next", 0)
    return 0


def _save_progress(scheme: str, next_index: int) -> None:
    _progress_path(scheme).write_text(
        json.dumps({"next": next_index}), encoding="utf-8"
    )


def _append_jsonl(path: Path, chunk: Chunk) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dataclasses.asdict(chunk), ensure_ascii=False) + "\n")


def _print_chunk(chunk: Chunk, index: int, total: int) -> None:
    """Pretty-print a chunk for review."""
    sep = "-" * 72
    typer.echo(f"\n{sep}")
    typer.echo(f"  Chunk {index + 1} / {total}")
    typer.echo(sep)
    if chunk.heading_path:
        typer.echo("  Path: " + " > ".join(chunk.heading_path))
    typer.echo(f"  Heading: {chunk.heading}")
    typer.echo(f"  Role:    {chunk.role}  |  Source: {chunk.source_document}")
    if chunk.page_range is not None:
        typer.echo(f"  Page:    {chunk.page_range}")
    typer.echo(
        f"  Tables: {len(chunk.tables)}  Figures: {len(chunk.figures)}  "
        f"Formulas: {len(chunk.formulas)}  Notes: {len(chunk.notes)}"
    )
    typer.echo(sep)
    body_preview = chunk.body[:500]
    if len(chunk.body) > 500:
        body_preview += " [...truncated]"
    typer.echo(body_preview)
    typer.echo(sep)


def review_scheme(scheme: str) -> None:
    """Interactively review chunks for *scheme*.

    Loads from ``data/chunks/<scheme>.jsonl`` and routes accepted/edited chunks
    to ``data/chunks/reviewed/<scheme>.jsonl``, skipped chunks to
    ``data/chunks/skipped/<scheme>.jsonl``.

    Progress is persisted so the session can be resumed with ``--review``.
    """
    chunks = load_chunks(scheme)
    total = len(chunks)
    if total == 0:
        typer.echo("No chunks found.")
        return

    reviewed_path = CHUNKS_DIR / _REVIEWED_DIR / f"{scheme.upper()}.jsonl"
    skipped_path = CHUNKS_DIR / _SKIPPED_DIR / f"{scheme.upper()}.jsonl"
    editor = os.environ.get("EDITOR", "notepad")

    start = _load_progress(scheme)
    if start >= total:
        typer.echo(
            f"All {total} chunks already reviewed. Delete the progress file to restart."
        )
        return
    if start > 0:
        typer.echo(f"Resuming from chunk {start + 1} / {total}")

    for i in range(start, total):
        chunk = chunks[i]
        _print_chunk(chunk, i, total)

        while True:
            raw = typer.prompt("[a]ccept  [e]dit  [s]kip  [q]uit").strip().lower()
            if raw in ("a", "e", "s", "q"):
                break
            typer.echo("  Please enter a, e, s, or q.")

        if raw == "q":
            _save_progress(scheme, i)
            typer.echo(f"\nProgress saved at chunk {i + 1}. Resume with --review {scheme}.")
            return

        elif raw == "a":
            _append_jsonl(reviewed_path, chunk)
            log.info("  Accepted chunk %d", i + 1)

        elif raw == "e":
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(json.dumps(dataclasses.asdict(chunk), indent=2, ensure_ascii=False))
                tmp_path = Path(tmp.name)
            try:
                subprocess.run([editor, str(tmp_path)], check=True)
                edited_text = tmp_path.read_text(encoding="utf-8")
                edited_dict = json.loads(edited_text)
                edited_chunk = Chunk(**edited_dict)
                _append_jsonl(reviewed_path, edited_chunk)
                log.info("  Accepted edited chunk %d", i + 1)
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                typer.echo(f"  Warning: edited JSON invalid ({exc}). Chunk NOT saved.")
            finally:
                tmp_path.unlink(missing_ok=True)

        elif raw == "s":
            _append_jsonl(skipped_path, chunk)
            log.info("  Skipped chunk %d", i + 1)

        _save_progress(scheme, i + 1)

    typer.echo(f"\nReview complete -- all {total} chunks processed.")
    _save_progress(scheme, total)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    scheme: Optional[str] = typer.Option(
        None,
        "--scheme",
        help="Chunk cleaned files for this scheme and write JSONL (Stage 3).",
    ),
    review: Optional[str] = typer.Option(
        None,
        "--review",
        metavar="SCHEME",
        help="Interactively review chunks for this scheme (Stage 4).",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print progress."),
) -> None:
    """Parse cleaned markdown into chunks (Stage 3) or review them (Stage 4)."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )

    if scheme is None and review is None:
        typer.echo("Error: provide --scheme <SCHEME> or --review <SCHEME>.", err=True)
        raise typer.Exit(code=1)
    if scheme is not None and review is not None:
        typer.echo("Error: --scheme and --review are mutually exclusive.", err=True)
        raise typer.Exit(code=1)

    # ---- Stage 3: chunk ----
    if scheme is not None:
        scheme = scheme.upper()
        log.info("Chunking cleaned files for scheme: %s", scheme)
        try:
            chunks = chunk_scheme(scheme)
        except FileNotFoundError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)

        out_path = CHUNKS_DIR / f"{scheme}.jsonl"
        save_chunks(chunks, out_path)
        typer.echo(f"Chunked {len(chunks)} block(s) -> {out_path}")
        return

    # ---- Stage 4: review ----
    if review is not None:
        review = review.upper()
        try:
            review_scheme(review)
        except FileNotFoundError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
