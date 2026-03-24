"""
query.py — Natural Language Reasoning Layer.

Sits on top of the existing vector store (ChromaDB) and structured database
(SQLite) and returns synthesised answers — not raw chunks.

Three query modes, routed automatically unless overridden:

SEMANTIC
    Retrieve top-k chunks from ChromaDB by semantic similarity, pass as
    context to GPT-4o, return a synthesised answer with source references.

STRUCTURED
    Query SQLite for exact values (benchmarks, thresholds, calculation
    methods). Match by scheme, credit_id, role, or keyword in body.

DOCUMENT
    Retrieve ALL chunks for a given credit_id or source_document from
    the JSONL chunks file and concatenate in document order. Pass full
    context to GPT-4o for a coherent summary.

CLI usage::

    python -m ratings.query "which criteria relate to material reduction?"
    python -m ratings.query "summarize criterion BNB 1.1.1" --scheme BNB
    python -m ratings.query "how many points for GWP?" --mode structured --scheme BNB
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import chromadb
import typer
from openai import OpenAI
from sqlalchemy import select, or_
from sqlalchemy.orm import Session

from ratings.config import (
    CHROMA_COLLECTION,
    CHUNKS_DIR,
    DB_PATH,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    QUERY_LLM_MODEL,
    QUERY_TOP_K,
    VECTORSTORE_DIR,
)
from ratings.database import get_engine
from ratings.schemas import CriteriaRecord

app = typer.Typer(
    name="query",
    help="Natural language reasoning layer over certification criteria.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Structured response from the query layer."""

    answer: str
    query_type: str
    sources: list[dict] = field(default_factory=list)
    confidence: str = "high"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert on building sustainability certification schemes "
    "including BNB, BREEAM, LEED, and DGNB. You answer questions strictly "
    "based on the provided context chunks. If the answer is not contained "
    "in the context, say so explicitly — do not infer or hallucinate values, "
    "especially benchmark thresholds or scoring criteria. Always cite the "
    "source criterion by its credit_id and heading when referencing specific "
    "requirements."
)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

# Patterns that suggest a specific credit or document
_CREDIT_ID_RE = re.compile(
    r"\b(?:BNB[_ ]?BN[_ ]?\d[\d.]*"
    r"|Mat[_ ]\d+"
    r"|Ene[_ ]\d+"
    r"|Man[_ ]\d+"
    r"|MRc\d+"
    r"|EAc\d+"
    r"|ENV\d[\d.]*"
    r"|ECO\d[\d.]*"
    r"|SOC\d[\d.]*"
    r"|TEC\d[\d.]*"
    r"|PRO\d[\d.]*"
    r")\b",
    re.IGNORECASE,
)
_DOC_NAME_RE = re.compile(
    r"\b(?:BNB_BN\d{4}_\d+\w*"
    r"|BREEAM_\w+"
    r"|LEED_\w+"
    r"|DGNB_\w+"
    r")\b",
    re.IGNORECASE,
)
_STRUCTURED_KEYWORDS = re.compile(
    r"\b(?:how many points|benchmark|threshold|scoring table"
    r"|Bewertungsmaßstab|Anforderungsniveau|Punkte|points"
    r"|required value|calculation method|what is the value"
    r"|wie viele Punkte|Grenzwert|Schwellenwert"
    r")\b",
    re.IGNORECASE,
)
_DOCUMENT_KEYWORDS = re.compile(
    r"\b(?:summarize|summary|zusammenfassung|zusammenfassen"
    r"|explain|overview|überblick|erklär|beschreib"
    r"|was ist Kriterium|what is criterion"
    r")\b",
    re.IGNORECASE,
)
# Bare numeric credit IDs like "1.1.1" or "2.3" that lack a scheme prefix
_NUMERIC_CREDIT_RE = re.compile(r"\b(\d+\.\d+(?:\.\d+)*)\b")

# Stop words filtered out during keyword extraction (English + German).
_STOP_WORDS: frozenset[str] = frozenset({
    # English
    "about", "after", "also", "back", "been", "both", "call", "come",
    "could", "does", "down", "each", "even", "every", "first", "from",
    "full", "give", "good", "hand", "have", "help", "into", "just",
    "know", "last", "less", "like", "long", "make", "many", "more",
    "most", "need", "only", "onto", "over", "part", "should", "some",
    "such", "take", "than", "that", "them", "then", "there", "these",
    "they", "this", "those", "time", "under", "very", "well", "were",
    "what", "when", "where", "which", "will", "with", "work", "would",
    "year",
    # German
    "aber", "alle", "allen", "also", "auch", "auf", "aus", "beim",
    "beide", "durch", "eine", "einem", "einen", "einer", "eines",
    "gegen", "haben", "hatte", "hier", "immer", "jede", "jedem",
    "jeden", "jeder", "jedes", "kann", "kein", "keine", "keinen",
    "keiner", "keines", "kommt", "macht", "mehr", "nach", "nicht",
    "noch", "oder", "ohne", "schon", "sehr", "sein", "seine",
    "sich", "sind", "soll", "über", "unter", "viel", "viele",
    "wann", "waren", "warum", "weil", "welche", "welchen", "welcher",
    "wenn", "wird", "wieder", "worden", "zum", "zur", "zwei",
})
# Abbreviation pattern: ALL-CAPS sequences (GWP, LCA, CO2), German mixed-case
# technical terms (EnEV, DIN), and eLCA-style lowercase-prefix forms.
_ABBREV_RE = re.compile(
    r"\b(?:[A-Z]{2,}[0-9]*|[A-Z][a-z]+[A-Z][a-zA-Z0-9]*|[a-z][A-Z][a-zA-Z]+)\b"
)


def route_query(question: str) -> str:
    """Classify *question* into ``semantic``, ``structured``, or ``document``.

    Priority (checked in order):
    1. Explicit credit ID or document name → ``document``
    2. Structured keywords (benchmarks, scores, thresholds) → ``structured``
    3. Everything else → ``semantic``
    """
    has_credit = bool(
        _CREDIT_ID_RE.search(question) or _NUMERIC_CREDIT_RE.search(question)
    )
    has_doc = bool(_DOC_NAME_RE.search(question))
    has_structured_kw = bool(_STRUCTURED_KEYWORDS.search(question))

    # Rules 1 & 2: explicit credit or document reference always wins
    if has_credit or has_doc:
        return "document"
    # Rule 3: numeric/benchmark questions
    if has_structured_kw:
        return "structured"
    return "semantic"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _synthesise(question: str, context_chunks: list[str]) -> str:
    """Call the LLM with *context_chunks* and return the answer text."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    context_block = "\n\n---\n\n".join(context_chunks)
    user_message = (
        f"Context chunks:\n\n{context_block}\n\n"
        f"---\n\nQuestion: {question}"
    )
    response = client.chat.completions.create(
        model=QUERY_LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------


def _embed_query(text: str) -> list[float]:
    """Embed *text* using OpenAI and return the embedding vector."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _semantic_retrieve(
    question: str,
    scheme: Optional[str] = None,
    top_k: int = QUERY_TOP_K,
) -> list[dict]:
    """Retrieve top-k chunks from ChromaDB by semantic similarity."""
    embedding = _embed_query(question)

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = client.get_collection(CHROMA_COLLECTION)

    query_kwargs: dict = {
        "query_embeddings": [embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if scheme:
        query_kwargs["where"] = {"scheme": scheme.upper()}

    results = collection.query(**query_kwargs)

    hits: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1.0 - dist / 2.0, 4)
        hits.append({
            "text": doc,
            "score": similarity,
            "credit_id": (meta or {}).get("credit_id", ""),
            "heading": (meta or {}).get("heading", ""),
            "source_document": (meta or {}).get("source_document", ""),
            "page_range": (meta or {}).get("page_number"),
        })
    return hits


def _structured_retrieve(
    question: str,
    scheme: Optional[str] = None,
) -> list[dict]:
    """Query SQLite for criteria matching keywords in the question."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            f"Run Stage 5 first:  python -m ratings.database --action init"
        )

    engine = get_engine()
    stmt = select(CriteriaRecord)

    if scheme:
        stmt = stmt.where(CriteriaRecord.scheme_name == scheme.upper())

    # Keyword filtering: look for significant words in the description
    words = [w for w in re.split(r"\s+", question) if len(w) > 3]
    if words:
        like_clauses = [CriteriaRecord.description.ilike(f"%{w}%") for w in words[:5]]
        stmt = stmt.where(or_(*like_clauses))

    stmt = stmt.order_by(CriteriaRecord.category, CriteriaRecord.credit_name)

    with Session(engine) as session:
        rows = session.scalars(stmt).all()

    return [
        {
            "text": (
                f"[{row.credit_name}] {row.category}\n"
                f"{row.description}\n"
                f"Benchmark: {row.benchmark_value or 'N/A'} {row.unit or ''}\n"
                f"Material reduction: {row.touches_material_reduction} | "
                f"Structural efficiency: {row.touches_structural_efficiency}"
            ),
            "credit_id": row.credit_name,
            "heading": row.category,
            "source_document": row.source_document,
            "page_range": row.page_number,
        }
        for row in rows
    ]


def _document_retrieve(
    question: str,
    scheme: Optional[str] = None,
    verbose: bool = False,
) -> list[dict]:
    """Retrieve all matching rows from SQLite for a credit_id or source_document.

    Strategies tried in strict priority order; each is only attempted when the
    previous tier returned zero results:

    1. Strict credit ID: question contains e.g. "BNB_BN 1.1.1" → ``credit_id``
    2. Document name: question contains e.g. "BNB_BN2015_111" → ``source_document``
    3. Numeric fallback: bare "1.1.1" → ``credit_id``; dots-stripped "111" →
       ``source_document``
    4. Credit-name keyword / abbreviation match: meaningful words (>4 chars,
       non-stop-word) searched in ``credit_name`` and ``category``; uppercase /
       mixed-case abbreviations (GWP, LCA, EnEV, eLCA) also searched in
       ``description``

    Results are sorted by heading depth (shallow first) then original insertion
    order so parent sections appear before their children.
    Never falls back to ChromaDB — returns an empty list when nothing matches.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            f"Run Stage 5 first:  python -m ratings.database --action init"
        )

    # --- Extract patterns from the question ---
    credit_match = _CREDIT_ID_RE.search(question)
    doc_match = _DOC_NAME_RE.search(question)
    numeric_match = _NUMERIC_CREDIT_RE.search(question)

    cid = credit_match.group(0) if credit_match else None
    doc = doc_match.group(0) if doc_match else None
    nid = numeric_match.group(1) if numeric_match else None
    nid_nodots = nid.replace(".", "") if nid else None

    # --- Verbose: show what we're searching for ---
    if verbose:
        typer.echo(
            f"[DEBUG] _document_retrieve patterns:"
            f" credit_id={cid!r}, doc_name={doc!r},"
            f" numeric={nid!r}, numeric_nodots={nid_nodots!r}"
        )
        engine_dbg = get_engine()
        with Session(engine_dbg) as s:
            samples = list(
                s.execute(
                    select(CriteriaRecord.credit_name).distinct().limit(10)
                ).scalars().all()
            )
        typer.echo(f"[DEBUG] Sample credit_name values in DB: {samples}")

    engine = get_engine()
    rows: list = []

    # -----------------------------------------------------------------------
    # Strategies 1–3: pattern-based (run as one OR-combined query)
    # -----------------------------------------------------------------------
    filter_conditions = []

    # Strategy 1 — strict credit ID → dedicated credit_id column
    if cid:
        filter_conditions.append(CriteriaRecord.credit_id.ilike(f"%{cid}%"))

    # Strategy 2 — explicit document name → source_document
    if doc:
        filter_conditions.append(CriteriaRecord.source_document.ilike(f"%{doc}%"))

    # Strategy 3 — numeric fallback (only when no strict credit-ID match)
    if nid and not credit_match:
        # 3a. dotted number in credit_id (scheme-agnostic)
        filter_conditions.append(CriteriaRecord.credit_id.ilike(f"%{nid}%"))
        # 3b. dots-stripped number in source_document (e.g. "111" in "BNB_BN2015_111")
        if nid_nodots:
            filter_conditions.append(
                CriteriaRecord.source_document.ilike(f"%{nid_nodots}%")
            )

    if filter_conditions:
        stmt = select(CriteriaRecord).order_by(CriteriaRecord.id)
        if scheme:
            stmt = stmt.where(CriteriaRecord.scheme_name == scheme.upper())
        stmt = stmt.where(or_(*filter_conditions))

        if verbose:
            from sqlalchemy.dialects import sqlite as _sa_sqlite
            compiled = stmt.compile(
                dialect=_sa_sqlite.dialect(),
                compile_kwargs={"literal_binds": True},
            )
            typer.echo(f"[DEBUG] SQL (strategies 1–3):\n{compiled}")

        with Session(engine) as session:
            rows = list(session.scalars(stmt).all())

    # -----------------------------------------------------------------------
    # Strategy 4: credit-name keyword + abbreviation match (last resort)
    # Only attempted when strategies 1–3 produced nothing.
    # -----------------------------------------------------------------------
    if not rows:
        # Extract abbreviations: ALL-CAPS (GWP, LCA, CO2), mixed-case (EnEV, eLCA)
        abbrevs = [m.group(0) for m in _ABBREV_RE.finditer(question)]
        # Extract substantive keywords: 5+ chars, not stop words, not abbreviations
        abbrev_set = set(abbrevs)
        kw_tokens = re.findall(r"\b[a-zA-Z\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\u00df]{5,}\b", question)
        keywords = [
            w.lower() for w in kw_tokens
            if w.lower() not in _STOP_WORDS and w not in abbrev_set
        ]

        if verbose:
            typer.echo(
                f"[DEBUG] Strategy 4 — abbrevs={abbrevs}, keywords={keywords}"
            )

        s4_conditions = []
        # Abbreviations: high-priority — match credit_name AND description
        for abbr in abbrevs:
            s4_conditions.append(CriteriaRecord.credit_name.ilike(f"%{abbr}%"))
            s4_conditions.append(CriteriaRecord.description.ilike(f"%{abbr}%"))
        # Keywords: match credit_name and category
        for kw in keywords:
            s4_conditions.append(CriteriaRecord.credit_name.ilike(f"%{kw}%"))
            s4_conditions.append(CriteriaRecord.category.ilike(f"%{kw}%"))

        if s4_conditions:
            stmt4 = select(CriteriaRecord).order_by(CriteriaRecord.id)
            if scheme:
                stmt4 = stmt4.where(CriteriaRecord.scheme_name == scheme.upper())
            stmt4 = stmt4.where(or_(*s4_conditions))

            with Session(engine) as session:
                rows = list(session.scalars(stmt4).all())

            if verbose:
                matched_names = sorted({r.credit_name for r in rows})
                typer.echo(
                    f"[DEBUG] credit_name matches found (strategy 4): {matched_names}"
                )

    # -----------------------------------------------------------------------
    # Warning when all strategies failed
    # -----------------------------------------------------------------------
    if not rows:
        typer.echo(
            f"[WARNING] _document_retrieve returned 0 rows.\n"
            f"  Strategies 1–3 searched: credit_id~={cid!r},"
            f" source_document~={doc!r}, numeric~={nid!r}/{nid_nodots!r}\n"
            f"  Scheme filter: {scheme!r}",
            err=True,
        )
        try:
            with Session(engine) as s:
                pairs = list(
                    s.execute(
                        select(
                            CriteriaRecord.credit_name,
                            CriteriaRecord.source_document,
                        )
                        .distinct()
                        .limit(10)
                    ).all()
                )
            typer.echo(
                f"[WARNING] Existing (credit_name, source_document) samples: {pairs}",
                err=True,
            )
        except Exception as _e:
            typer.echo(f"[WARNING] Could not query DB for samples: {_e}", err=True)
        return []

    # --- Sort: shallower category depth first, then by original insertion order ---
    def _depth_key(row: CriteriaRecord) -> tuple:
        depth = row.category.count(".") if row.category else 0
        return (depth, row.id)

    sorted_rows = sorted(rows, key=_depth_key)

    return [
        {
            "text": (
                f"### [{row.credit_name}] {row.category}\n\n"
                f"{row.description}"
                + (
                    f"\n\nBenchmark/Threshold: {row.benchmark_value} "
                    f"{row.unit or ''}".rstrip()
                    if row.benchmark_value
                    else ""
                )
            ),
            "credit_id": row.credit_name,
            "heading": row.category,
            "source_document": row.source_document,
            "page_range": row.page_number,
        }
        for row in sorted_rows
    ]


# ---------------------------------------------------------------------------
# Main query function
# ---------------------------------------------------------------------------


def query(
    question: str,
    mode: str = "auto",
    scheme: Optional[str] = None,
    top_k: int = QUERY_TOP_K,
    verbose: bool = False,
) -> QueryResult:
    """Answer *question* using the appropriate retrieval + synthesis pipeline.

    Parameters
    ----------
    question:
        Natural language query.
    mode:
        ``"auto"`` routes based on heuristics. Override with
        ``"semantic"``, ``"structured"``, or ``"document"``.
    scheme:
        If provided, restrict results to this certification scheme.
    top_k:
        Number of chunks for semantic retrieval.
    verbose:
        If *True*, print retrieved chunks before synthesis.

    Returns
    -------
    QueryResult
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )

    # Route — explicit mode bypasses the router entirely
    explicit_mode = mode != "auto"
    query_type = mode if explicit_mode else route_query(question)

    # Retrieve — fallback to semantic only when mode was auto-routed
    if query_type == "semantic":
        hits = _semantic_retrieve(question, scheme=scheme, top_k=top_k)
    elif query_type == "structured":
        hits = _structured_retrieve(question, scheme=scheme)
        if not hits and not explicit_mode:
            hits = _semantic_retrieve(question, scheme=scheme, top_k=top_k)
            query_type = "semantic"
    elif query_type == "document":
        hits = _document_retrieve(question, scheme=scheme, verbose=verbose)
        if not hits and not explicit_mode:
            hits = _semantic_retrieve(question, scheme=scheme, top_k=top_k)
            query_type = "semantic"
    else:
        raise ValueError(f"Unknown query mode: {mode!r}")

    if not hits:
        return QueryResult(
            answer="No relevant chunks found for this query.",
            query_type=query_type,
            sources=[],
            confidence="low",
        )

    # Verbose: print retrieved chunks
    if verbose:
        typer.echo(f"\n--- Retrieved {len(hits)} chunk(s) [{query_type}] ---")
        for i, h in enumerate(hits, 1):
            preview = h["text"][:300]
            typer.echo(f"\n[{i}] {h.get('credit_id', '?')} | {h.get('heading', '?')}")
            typer.echo(f"    {preview}{'...' if len(h['text']) > 300 else ''}")
        typer.echo("--- End retrieved chunks ---\n")

    # Synthesise
    context_chunks = [h["text"] for h in hits]
    answer = _synthesise(question, context_chunks)

    # Build deduplicated sources list
    seen: set[tuple] = set()
    sources: list[dict] = []
    for h in hits:
        key = (h.get("credit_id"), h.get("heading"), h.get("source_document"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "credit_id": h.get("credit_id", ""),
                "heading": h.get("heading", ""),
                "source_document": h.get("source_document", ""),
                "page_range": h.get("page_range"),
            })

    # Confidence: high if all hits are from exact matches or high similarity
    confidence = "high"
    if query_type == "semantic":
        scores = [h.get("score", 0) for h in hits]
        if scores and max(scores) < 0.75:
            confidence = "low"

    return QueryResult(
        answer=answer,
        query_type=query_type,
        sources=sources,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    question: str = typer.Argument(..., help="Natural language question."),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help="Query mode: auto, semantic, structured, or document.",
    ),
    scheme: Optional[str] = typer.Option(
        None,
        "--scheme",
        help="Filter results to this certification scheme (e.g. BNB).",
    ),
    top_k: int = typer.Option(
        QUERY_TOP_K,
        "--top-k",
        help="Number of chunks for semantic retrieval.",
        min=1,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose/--quiet",
        help="Print retrieved chunks before the answer.",
    ),
) -> None:
    """Ask a question about certification criteria."""
    try:
        result = query(
            question=question,
            mode=mode,
            scheme=scheme,
            top_k=top_k,
            verbose=verbose,
        )
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    # Print result
    typer.echo(f"\n{'═' * 72}")
    typer.echo(f"  Mode: {result.query_type}  |  Confidence: {result.confidence}")
    typer.echo(f"{'═' * 72}\n")
    typer.echo(result.answer)
    typer.echo(f"\n{'─' * 72}")
    typer.echo("  Sources:")
    for s in result.sources:
        parts = []
        if s["credit_id"]:
            parts.append(s["credit_id"])
        if s["heading"]:
            parts.append(s["heading"])
        if s["source_document"]:
            parts.append(f"[{s['source_document']}]")
        if s.get("page_range") is not None:
            parts.append(f"p.{s['page_range']}")
        typer.echo(f"    - {' | '.join(parts)}")
    typer.echo(f"{'─' * 72}")


if __name__ == "__main__":
    app()
