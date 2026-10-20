# RatingSystems — Sustainability Certification Knowledge System

A modular Python pipeline to extract, structure, and query information from PDF manuals of building certification schemes (BREEAM, LEED, DGNB, BNB), powering a consulting and advisory tool.

---

## Architecture

```
PDFs
 │
 ▼
Stage 1 — Ingest        pymupdf4llm → Markdown files  (data/raw/)
 │
 ▼
Stage 2 — Chunk         LlamaIndex HierarchicalNodeParser → serialized nodes  (data/processed/)
 │
 ▼
Stage 3 — Vectorstore   OpenAI text-embedding-3-large → ChromaDB  (data/vectorstore/)
 │
 ▼
Stage 4 — Extract       LLM + instructor → CertificationCriteria Pydantic objects → JSON
 │
 ▼
Stage 5 — Database      SQLAlchemy ORM → SQLite  (data/ratings.db)
 │
 ▼
Stage 6 — Query         Semantic search (ChromaDB) + Structured filters (SQLite)
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the pipeline

Each stage is an independent CLI module. Run them in order for a new scheme/document set.

---

## Stage Reference

### Stage 1 — PDF Ingestion

```bash
# Ingest a single PDF
python -m ratings.ingest --pdf path/to/manual.pdf --scheme BREEAM

# Ingest all PDFs in a directory
python -m ratings.ingest --pdf-dir path/to/pdfs/ --scheme LEED
```

Output: `data/raw/{scheme}/{document_stem}.md`

---

### Stage 2 — Chunking + Embedding Preparation

```bash
python -m ratings.chunk --scheme BREEAM
```

Reads all `.md` files from `data/raw/{scheme}/`, applies `HierarchicalNodeParser` (chunk sizes: 2048 / 512 / 128 tokens), and serialises nodes to JSON.

Output: `data/processed/{scheme}/nodes.json`

---

### Stage 3 — Vector Store (ChromaDB)

```bash
python -m ratings.vectorstore --scheme BREEAM
```

Loads processed nodes, generates OpenAI `text-embedding-3-large` embeddings, and upserts into a local ChromaDB persistent store. All chunks from all schemes share one collection (`certification_criteria`) with a `scheme` metadata field for filtering.

Output: `data/vectorstore/` (ChromaDB files)

---

### Stage 4 — Structured Extraction

```bash
python -m ratings.extract --scheme BREEAM
python -m ratings.extract --scheme BREEAM --limit 20   # cap at 20 chunks (useful for testing)
```

Queries ChromaDB for chunks belonging to `{scheme}`, then uses an OpenAI LLM (via `instructor`) to extract structured `CertificationCriteria` objects from each chunk.

Output: `data/processed/{scheme}/criteria.json`

---

### Stage 5 — Relational Database

```bash
# Create the SQLite database and tables (run once)
python -m ratings.database --action init

# Load extracted criteria into the database
python -m ratings.database --action load --scheme BREEAM
```

Output: `data/ratings.db` (SQLite)

---

### Stage 6 — Query

```bash
# Semantic search across all schemes
python -m ratings.query --semantic "embodied carbon reduction"

# Semantic search filtered to one scheme, top 10 results
python -m ratings.query --semantic "structural efficiency" --scheme DGNB --top-k 10

# Structured filter: all criteria that touch material reduction
python -m ratings.query --structured --material

# Structured filter: LEED criteria touching structural efficiency
python -m ratings.query --structured --scheme LEED --structural

# Combine flags
python -m ratings.query --structured --scheme BREEAM --material --structural
```

---

## Data Schema

### `CertificationCriteria` (Pydantic)

| Field | Type | Description |
|---|---|---|
| `scheme_name` | `str` | BREEAM / LEED / DGNB / BNB |
| `category` | `str` | e.g. "Materials", "Energy" |
| `credit_name` | `str` | Name of the credit/criterion |
| `description` | `str` | Full text description |
| `benchmark_value` | `str \| None` | Numeric or qualitative benchmark |
| `unit` | `str \| None` | Unit of benchmark |
| `touches_material_reduction` | `bool` | Criterion relates to material efficiency / reduction |
| `touches_structural_efficiency` | `bool` | Criterion relates to structural efficiency |
| `source_document` | `str` | Original PDF filename |
| `page_number` | `int \| None` | Source page |

---

## Configuration

All paths and model settings are in `ratings/config.py`. Environment variables override defaults:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `LLM_MODEL` | `gpt-4o` | OpenAI model for extraction |

---

## Project Structure

```
RatingSystems/
├── data/
│   ├── raw/            # Markdown extracted from PDFs
│   ├── processed/      # Serialised nodes + criteria JSON
│   └── vectorstore/    # ChromaDB persistent files
├── ratings/
│   ├── __init__.py
│   ├── config.py       # Paths and model settings
│   ├── schemas.py      # Pydantic + SQLAlchemy models
│   ├── ingest.py       # Stage 1 — PDF → Markdown
│   ├── chunk.py        # Stage 2 — Chunking
│   ├── vectorstore.py  # Stage 3 — Embed + store
│   ├── extract.py      # Stage 4 — LLM extraction
│   ├── database.py     # Stage 5 — SQLite persistence
│   └── query.py        # Stage 6 — Query interface
├── tests/
│   └── test_smoke.py
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
```

Smoke tests verify imports, schema validation, and ORM instantiation without requiring API keys or real data.

---

## Working with AI agents

AI agents working in this repo must follow [`CLAUDE.md`](CLAUDE.md). Most importantly: **nothing gets deleted without Alina's written OK in the chat**; an "allow" click in a permission dialog does not count.
