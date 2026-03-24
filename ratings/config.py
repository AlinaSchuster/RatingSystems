"""
config.py — Central configuration for paths and model settings.

All pipeline stages import from here.  Environment variables override
default values where noted.  The .env file at the project root is loaded
automatically via python-dotenv.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root: the directory that contains this package (ratings/)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load .env from project root (silently ignored if the file is absent)
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"

#: Stage 1 output — raw markdown extracted from PDFs
RAW_DIR: Path = DATA_DIR / "raw"

#: Stage 2 output — serialised LlamaIndex nodes (JSON) and Stage 4 JSON
PROCESSED_DIR: Path = DATA_DIR / "processed"

#: Stage 3 output — ChromaDB persistent storage
VECTORSTORE_DIR: Path = DATA_DIR / "vectorstore"

#: Stage 5 output — SQLite database file
DB_PATH: Path = DATA_DIR / "ratings.db"

#: Inbox — drop PDFs here; ingest will pick them up and move them to DONE_DIR
INBOX_DIR: Path = DATA_DIR / "inbox"

#: Done — PDFs are moved here automatically after successful ingestion
DONE_DIR: Path = DATA_DIR / "done"

#: Stage 2 output — human-edited/validated markdown files
CLEANED_DIR: Path = DATA_DIR / "cleaned"

#: Stage 3 output — semantic chunks in JSONL format
CHUNKS_DIR: Path = DATA_DIR / "chunks"

# ---------------------------------------------------------------------------
# OpenAI / LLM settings
# ---------------------------------------------------------------------------
#: OpenAI API key — must be set in the environment or .env file
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

#: Embedding model for Stage 2/3
EMBEDDING_MODEL: str = "text-embedding-3-large"

#: LLM for Stage 4 structured extraction — override via LLM_MODEL env var
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o")

# ---------------------------------------------------------------------------
# Query layer settings
# ---------------------------------------------------------------------------
#: Default top-k chunks for semantic retrieval in query layer
QUERY_TOP_K: int = int(os.environ.get("QUERY_TOP_K", "5"))

#: LLM for query synthesis — override via QUERY_LLM_MODEL env var
QUERY_LLM_MODEL: str = os.environ.get("QUERY_LLM_MODEL", "gpt-4o")

# ---------------------------------------------------------------------------
# ChromaDB settings
# ---------------------------------------------------------------------------
#: Single collection name; use ``scheme`` metadata field to filter per scheme
CHROMA_COLLECTION: str = "certification_criteria"

# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------
#: Hierarchical chunk sizes (largest → smallest) in tokens
CHUNK_SIZES: list[int] = [2048, 512, 128]

# ---------------------------------------------------------------------------
# Ensure data directories exist at import time
# ---------------------------------------------------------------------------
for _dir in (RAW_DIR, PROCESSED_DIR, VECTORSTORE_DIR, INBOX_DIR, DONE_DIR, CLEANED_DIR, CHUNKS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
