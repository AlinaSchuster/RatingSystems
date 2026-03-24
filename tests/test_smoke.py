"""
test_smoke.py — Lightweight smoke tests.

These tests verify that modules can be imported, schemas validate correctly,
and ORM models can be instantiated — all without requiring API keys, real
PDFs, or a running database.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_imports() -> None:
    """config.py should import cleanly and expose expected attributes."""
    from ratings import config  # noqa: PLC0415

    assert hasattr(config, "PROJECT_ROOT"), "PROJECT_ROOT missing"
    assert hasattr(config, "RAW_DIR"), "RAW_DIR missing"
    assert hasattr(config, "PROCESSED_DIR"), "PROCESSED_DIR missing"
    assert hasattr(config, "VECTORSTORE_DIR"), "VECTORSTORE_DIR missing"
    assert hasattr(config, "DB_PATH"), "DB_PATH missing"
    assert isinstance(config.PROJECT_ROOT, Path)
    assert isinstance(config.CHUNK_SIZES, list)
    assert len(config.CHUNK_SIZES) == 3


def test_config_data_dirs_exist() -> None:
    """Data directories should be created automatically on import."""
    from ratings.config import PROCESSED_DIR, RAW_DIR, VECTORSTORE_DIR  # noqa: PLC0415

    assert RAW_DIR.exists(), f"RAW_DIR not created: {RAW_DIR}"
    assert PROCESSED_DIR.exists(), f"PROCESSED_DIR not created: {PROCESSED_DIR}"
    assert VECTORSTORE_DIR.exists(), f"VECTORSTORE_DIR not created: {VECTORSTORE_DIR}"


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

VALID_CRITERION: dict = {
    "scheme_name": "BREEAM",
    "category": "Materials",
    "credit_name": "Mat 01",
    "description": "Life cycle impacts of construction materials.",
    "benchmark_value": "10",
    "unit": "points",
    "touches_material_reduction": True,
    "touches_structural_efficiency": False,
    "source_document": "BREEAM_NC_2018.pdf",
    "page_number": 123,
}


def test_certification_criteria_valid() -> None:
    """CertificationCriteria should accept a fully populated valid dict."""
    from ratings.schemas import CertificationCriteria  # noqa: PLC0415

    obj = CertificationCriteria(**VALID_CRITERION)
    assert obj.scheme_name == "BREEAM"
    assert obj.touches_material_reduction is True
    assert obj.page_number == 123


def test_certification_criteria_optional_fields() -> None:
    """CertificationCriteria should accept None for optional fields."""
    from ratings.schemas import CertificationCriteria  # noqa: PLC0415

    minimal = {
        "scheme_name": "LEED",
        "category": "Materials & Resources",
        "credit_name": "MRc1",
        "description": "Building life-cycle impact reduction.",
        "touches_material_reduction": True,
        "touches_structural_efficiency": False,
        "source_document": "LEED_v4_BD+C.pdf",
    }
    obj = CertificationCriteria(**minimal)
    assert obj.benchmark_value is None
    assert obj.unit is None
    assert obj.page_number is None


def test_certification_criteria_model_dump() -> None:
    """model_dump() should round-trip back to the original dict."""
    from ratings.schemas import CertificationCriteria  # noqa: PLC0415

    obj = CertificationCriteria(**VALID_CRITERION)
    dumped = obj.model_dump()
    assert dumped["scheme_name"] == VALID_CRITERION["scheme_name"]
    assert dumped["credit_name"] == VALID_CRITERION["credit_name"]


# ---------------------------------------------------------------------------
# SQLAlchemy ORM
# ---------------------------------------------------------------------------


def test_criteria_record_from_pydantic() -> None:
    """CriteriaRecord.from_pydantic() should produce a valid ORM instance."""
    from ratings.schemas import CertificationCriteria, CriteriaRecord  # noqa: PLC0415

    pydantic_obj = CertificationCriteria(**VALID_CRITERION)
    record = CriteriaRecord.from_pydantic(pydantic_obj)

    assert record.scheme_name == "BREEAM"
    assert record.touches_material_reduction is True
    assert record.page_number == 123


def test_criteria_record_to_dict() -> None:
    """to_dict() should not include the internal ``id`` field."""
    from ratings.schemas import CertificationCriteria, CriteriaRecord  # noqa: PLC0415

    pydantic_obj = CertificationCriteria(**VALID_CRITERION)
    record = CriteriaRecord.from_pydantic(pydantic_obj)
    d = record.to_dict()

    assert "id" not in d
    assert d["scheme_name"] == "BREEAM"
    assert d["category"] == "Materials"


def test_orm_table_name() -> None:
    """CriteriaRecord should map to the 'criteria' table."""
    from ratings.schemas import CriteriaRecord  # noqa: PLC0415

    assert CriteriaRecord.__tablename__ == "criteria"


# ---------------------------------------------------------------------------
# Module imports (all pipeline stages)
# ---------------------------------------------------------------------------


def test_all_stage_modules_import() -> None:
    """All pipeline stage modules should import without errors."""
    import ratings.ingest  # noqa: F401, PLC0415
    import ratings.clean  # noqa: F401, PLC0415
    import ratings.chunk  # noqa: F401, PLC0415
    import ratings.vectorstore  # noqa: F401, PLC0415
    import ratings.extract  # noqa: F401, PLC0415
    import ratings.database  # noqa: F401, PLC0415
    import ratings.query  # noqa: F401, PLC0415


def test_stage_apps_are_typer_instances() -> None:
    """Each stage module should expose a Typer ``app`` instance."""
    import typer  # noqa: PLC0415

    from ratings import ingest, clean, chunk, vectorstore, extract, database, query  # noqa: PLC0415

    for module in (ingest, clean, chunk, vectorstore, extract, database, query):
        assert isinstance(module.app, typer.Typer), (
            f"{module.__name__}.app is not a Typer instance"
        )


# ---------------------------------------------------------------------------
# Stage 2 — clean module
# ---------------------------------------------------------------------------


def test_config_cleaned_and_chunks_dirs_exist() -> None:
    """CLEANED_DIR and CHUNKS_DIR should be created on import."""
    from ratings.config import CLEANED_DIR, CHUNKS_DIR  # noqa: PLC0415

    assert CLEANED_DIR.exists(), f"CLEANED_DIR not found: {CLEANED_DIR}"
    assert CHUNKS_DIR.exists(), f"CHUNKS_DIR not found: {CHUNKS_DIR}"


def test_clean_module_imports() -> None:
    """ratings.clean should import without errors."""
    import ratings.clean  # noqa: F401, PLC0415


def test_clean_validate_missing_frontmatter(tmp_path: "Path") -> None:
    """validate_file should return a violation when frontmatter is absent."""
    from ratings.clean import validate_file  # noqa: PLC0415

    md = tmp_path / "test.md"
    md.write_text("## Some Heading [credit]\n\nsome text\n", encoding="utf-8")
    violations = validate_file(md)
    assert any("frontmatter" in v.lower() for v in violations), (
        f"Expected a frontmatter violation, got: {violations}"
    )


def test_clean_validate_valid_file(tmp_path: "Path") -> None:
    """A properly formatted file should produce no violations."""
    from ratings.clean import validate_file  # noqa: PLC0415

    content = (
        "---\n"
        "scheme: TEST\n"
        "version: '1.0'\n"
        "document: Test Document\n"
        "content_type: credits\n"
        "---\n"
        "\n"
        "## Category A [category]\n"
        "\n"
        "### Credit 1 [credit]\n"
        "\n"
        "Some descriptive text.\n"
    )
    md = tmp_path / "valid.md"
    md.write_text(content, encoding="utf-8")
    violations = validate_file(md)
    assert violations == [], f"Expected no violations, got: {violations}"


def test_clean_validate_heading_level_skip(tmp_path: "Path") -> None:
    """Skipping heading levels (H2 -> H4) should produce a violation."""
    from ratings.clean import validate_file  # noqa: PLC0415

    content = (
        "---\n"
        "scheme: TEST\n"
        "version: '1.0'\n"
        "document: Test Document\n"
        "content_type: credits\n"
        "---\n"
        "\n"
        "## Category A [category]\n"
        "\n"
        "#### Sub-sub [criterion]\n"
    )
    md = tmp_path / "skip.md"
    md.write_text(content, encoding="utf-8")
    violations = validate_file(md)
    assert any("level" in v.lower() or "skip" in v.lower() for v in violations), (
        f"Expected a heading-level violation, got: {violations}"
    )


# ---------------------------------------------------------------------------
# Stage 3 — chunk module
# ---------------------------------------------------------------------------


def test_chunk_module_imports() -> None:
    """ratings.chunk should import without errors."""
    import ratings.chunk  # noqa: F401, PLC0415


def test_chunk_dataclass_fields() -> None:
    """Chunk dataclass should have all required fields."""
    import dataclasses  # noqa: PLC0415

    from ratings.chunk import Chunk  # noqa: PLC0415

    field_names = {f.name for f in dataclasses.fields(Chunk)}
    required = {
        "scheme", "version", "content_type",
        "heading_path", "heading", "role", "body",
        "tables", "figures", "formulas", "notes",
        "source_document", "page_range",
    }
    missing = required - field_names
    assert not missing, f"Chunk is missing fields: {missing}"


def test_parse_chunks_simple(tmp_path: "Path") -> None:
    """A minimal cleaned file should produce one Chunk with the expected role."""
    from ratings.chunk import parse_chunks  # noqa: PLC0415

    content = (
        "---\n"
        "scheme: TEST\n"
        "version: '1.0'\n"
        "document: Minimal\n"
        "content_type: credits\n"
        "---\n"
        "\n"
        "## My Credit [credit]\n"
        "\n"
        "Applicants shall demonstrate compliance.\n"
    )
    md = tmp_path / "minimal.md"
    md.write_text(content, encoding="utf-8")
    chunks = parse_chunks(md)

    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert chunks[0].role == "credit"
    assert chunks[0].heading == "My Credit [credit]"
    assert "compliance" in chunks[0].body

