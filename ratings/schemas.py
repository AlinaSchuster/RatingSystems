"""
schemas.py — Pydantic and SQLAlchemy data models.

Both representations share the same field names so that converting between
them is a simple dict round-trip.

Pydantic
--------
``CertificationCriteria`` — validated data model used during extraction (Stage 4)
and as the interchange format between stages.

SQLAlchemy
----------
``CriteriaRecord`` — ORM model persisted to SQLite (Stage 5).  Indexed on
``scheme_name``, ``touches_material_reduction``, and
``touches_structural_efficiency`` for fast structured queries (Stage 6).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class CertificationCriteria(BaseModel):
    """A single extractable criterion from a building certification manual."""

    scheme_name: str = Field(
        description="Name of the certification scheme, e.g. BREEAM, LEED, DGNB, BNB."
    )
    category: str = Field(
        description="Top-level category or chapter the criterion belongs to, "
        "e.g. 'Materials', 'Energy', 'Land Use'."
    )
    credit_name: str = Field(
        description="Short name or code of the credit/criterion, e.g. 'Mat 01'."
    )
    credit_id: Optional[str] = Field(
        default=None,
        description="Scheme-specific unique identifier for the criterion as it appears "
        "in the source document, e.g. 'BNB_BN 1.1.1', 'Mat 01', 'EAc1'."
    )
    version: Optional[str] = Field(
        default=None,
        description="Document version or edition identifier, e.g. 'BN2015'.",
    )
    content_type: Optional[str] = Field(
        default=None,
        description="Type of content in the source file, e.g. 'credits', 'guidance'.",
    )
    extra_metadata: Optional[str] = Field(
        default=None,
        description="JSON string capturing any additional frontmatter fields not "
        "covered by dedicated model fields. Preserves scheme-specific metadata.",
    )
    building_type: Optional[str] = Field(
        default=None,
        description="Building type the criterion applies to, "
        "e.g. 'Büro- und Verwaltungsgebäude'.",
    )
    description: str = Field(
        description="Full description of the criterion as found in the manual."
    )
    benchmark_value: Optional[str] = Field(
        default=None,
        description="Numeric or qualitative threshold required to achieve the credit.",
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit for the benchmark value, e.g. 'kg CO2e/m²', '%', 'points'.",
    )
    touches_material_reduction: bool = Field(
        description="True if the criterion directly incentivises reducing material "
        "quantities, embodied carbon, or material efficiency."
    )
    touches_structural_efficiency: bool = Field(
        description="True if the criterion directly incentivises structural "
        "efficiency, load-bearing optimisation, or structural material reduction."
    )
    source_document: str = Field(
        description="Filename of the source PDF from which this criterion was extracted."
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Page number in the source document where the criterion appears.",
    )


# ---------------------------------------------------------------------------
# SQLAlchemy ORM model
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class CriteriaRecord(Base):
    """ORM representation of ``CertificationCriteria`` stored in SQLite."""

    __tablename__ = "criteria"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scheme_name: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str] = mapped_column(String(256), nullable=False)
    credit_name: Mapped[str] = mapped_column(String(256), nullable=False)
    credit_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    content_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    extra_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    building_type: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    benchmark_value: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    unit: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    touches_material_reduction: Mapped[bool] = mapped_column(Boolean, nullable=False)
    touches_structural_efficiency: Mapped[bool] = mapped_column(Boolean, nullable=False)
    source_document: Mapped[str] = mapped_column(String(512), nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Composite index for the most common structured-query patterns
    __table_args__ = (
        Index("ix_criteria_scheme", "scheme_name"),
        Index(
            "ix_criteria_relevance",
            "scheme_name",
            "touches_material_reduction",
            "touches_structural_efficiency",
        ),
    )

    @classmethod
    def from_pydantic(cls, obj: CertificationCriteria) -> "CriteriaRecord":
        """Create an ORM record from a validated Pydantic instance."""
        return cls(**obj.model_dump())

    def to_dict(self) -> dict:
        """Return a plain dictionary representation (excludes internal ``id``)."""
        return {
            "scheme_name": self.scheme_name,
            "category": self.category,
            "credit_name": self.credit_name,
            "credit_id": self.credit_id,
            "version": self.version,
            "content_type": self.content_type,
            "description": self.description,
            "benchmark_value": self.benchmark_value,
            "unit": self.unit,
            "touches_material_reduction": self.touches_material_reduction,
            "touches_structural_efficiency": self.touches_structural_efficiency,
            "source_document": self.source_document,
            "page_number": self.page_number,
            "extra_metadata": self.extra_metadata,
            "building_type": self.building_type,
        }
