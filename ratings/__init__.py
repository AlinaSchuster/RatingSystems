"""
ratings — Sustainability Certification Knowledge System.

Modular pipeline:
  Stage 1  ratings.ingest       PDF  -> Markdown (raw extraction)
  Stage 2  ratings.clean        Markdown cleaning & validation
  Stage 3  ratings.chunk        Cleaned Markdown -> JSONL chunks
  Stage 4  ratings.chunk        (--review) Interactive chunk review
  Stage 5  ratings.vectorstore  Nodes -> ChromaDB
  Stage 6  ratings.extract      Chunks -> Pydantic criteria (LLM)
  Stage 7  ratings.database     Criteria -> SQLite
  Stage 8  ratings.query        Semantic search + structured filters
"""
