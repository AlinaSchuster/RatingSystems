"""
vectorstore.py — Stage 3: Embedding + ChromaDB Vector Store.

Loads serialised LlamaIndex nodes for a scheme (produced by Stage 2),
generates OpenAI ``text-embedding-3-large`` embeddings, and upserts them
into a local ChromaDB persistent collection.  All schemes share a single
collection (``certification_criteria``); chunks are tagged with a ``scheme``
metadata field so they can be filtered per-scheme at query time (Stage 6).

CLI usage
---------
python -m ratings.vectorstore --scheme BREEAM
"""

from __future__ import annotations

import typer
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from ratings.config import (
    CHROMA_COLLECTION,
    CHUNKS_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    VECTORSTORE_DIR,
)

app = typer.Typer(
    name="vectorstore",
    help="Stage 3 — Embed nodes and upsert into ChromaDB.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _load_nodes(scheme: str):
    """Build LlamaIndex TextNode objects from the JSONL chunks file (Stage 3 output)."""
    import json as _json
    from llama_index.core.schema import TextNode

    jsonl_path = CHUNKS_DIR / f"{scheme}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {jsonl_path}\n"
            f"Run Stage 3 first:  python -m ratings.chunk --scheme {scheme}"
        )

    nodes = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = _json.loads(line)
            node_id = f"{record.get('source_document', 'unknown')}::{record.get('heading', '')}"
            metadata = {
                "scheme": record.get("scheme", scheme),
                "source_document": record.get("source_document", "unknown"),
                "heading": record.get("heading", ""),
                "section_context": " > ".join(record.get("heading_path", [])),
                "role": record.get("role", ""),
                "credit_id": record.get("credit_id") or "",
                "credit_name": record.get("credit_name") or "",
                "category": record.get("category") or "",
            }
            nodes.append(TextNode(
                id_=node_id,
                text=record.get("body", ""),
                metadata=metadata,
            ))
    return nodes


def get_chroma_collection() -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection.

    Returns
    -------
    chromadb.Collection
        The ``certification_criteria`` collection backed by the local
        persistent store at ``data/vectorstore/``.
    """
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def embed_and_store(scheme: str, *, verbose: bool = True) -> int:
    """Embed nodes for *scheme* and upsert them into ChromaDB.

    Uses a ``VectorStoreIndex`` backed by ``ChromaVectorStore`` so that
    LlamaIndex handles batching, retries, and rate-limit back-off automatically.

    Parameters
    ----------
    scheme:
        Certification scheme identifier (e.g. ``"BREEAM"``).
    verbose:
        If *True*, print progress messages.

    Returns
    -------
    int
        Total number of documents now in the ChromaDB collection (across all
        schemes).

    Raises
    ------
    FileNotFoundError
        If no serialised nodes exist for *scheme* (Stage 2 not yet run).
    ValueError
        If ``OPENAI_API_KEY`` is not configured.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set.  Add it to your .env file or environment."
        )

    if verbose:
        typer.echo(f"Loading nodes for scheme: {scheme}")

    nodes = _load_nodes(scheme)

    if verbose:
        typer.echo(f"  {len(nodes)} leaf node(s) loaded")
        typer.echo("Building embeddings and upserting into ChromaDB …")

    embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    collection = get_chroma_collection()

    # Remove any previously stored chunks for this scheme so re-runs don't
    # accumulate stale entries when node IDs change (e.g. after re-ingestion).
    existing = collection.get(where={"scheme": scheme}, include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        if verbose:
            typer.echo(f"  Removed {len(existing['ids'])} stale entry(ies) for scheme '{scheme}'")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # VectorStoreIndex.from_documents handles embed-and-upsert in one call.
    # show_progress requires tqdm, so we gate it on verbose.
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=verbose,
    )

    total = collection.count()
    return total


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
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print progress."),
) -> None:
    """Embed nodes and upsert into ChromaDB (Stage 3)."""
    scheme = scheme.upper()

    try:
        total = embed_and_store(scheme, verbose=verbose)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    if verbose:
        typer.echo(f"\nVector store update complete for scheme: {scheme}")
        typer.echo(f"Total documents in collection '{CHROMA_COLLECTION}': {total:,}")


if __name__ == "__main__":
    app()
