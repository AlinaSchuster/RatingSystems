"""
database.py — Stage 5: Relational Database (SQLite via SQLAlchemy).

Provides two actions:

``init``
    Create the SQLite database file and all ORM tables at
    ``data/ratings.db``.  Safe to run multiple times (uses
    ``checkfirst=True``).

``load``
    Read ``data/processed/{scheme}/criteria.json`` (produced by Stage 4)
    and upsert the criteria into the ``criteria`` table.  Existing rows
    for the scheme are replaced to avoid duplicates on re-runs.

CLI usage
---------
python -m ratings.database --action init
python -m ratings.database --action load --scheme BREEAM
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import typer
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from ratings.config import DB_PATH
from ratings.extract import load_criteria
from ratings.schemas import Base, CriteriaRecord

app = typer.Typer(
    name="database",
    help="Stage 5 — Persist structured criteria to SQLite.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Action(str, Enum):
    init = "init"
    load = "load"


def get_engine():
    """Return a SQLAlchemy engine connected to the project SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}", echo=False)


def init_db() -> None:
    """Create all ORM tables in the SQLite database.

    Uses ``checkfirst=True`` so existing tables are not dropped or modified.
    """
    engine = get_engine()
    Base.metadata.create_all(engine, checkfirst=True)


def load_scheme(scheme: str, *, verbose: bool = True) -> int:
    """Load extracted criteria for *scheme* into the SQLite database.

    Existing rows for *scheme* are deleted before inserting new ones so that
    re-running after a Stage 4 re-extraction produces a clean, consistent
    state.

    Parameters
    ----------
    scheme:
        Certification scheme identifier (e.g. ``"BREEAM"``).
    verbose:
        If *True*, print progress messages.

    Returns
    -------
    int
        Number of rows inserted.

    Raises
    ------
    FileNotFoundError
        If the criteria JSON file for *scheme* does not exist.
    """
    criteria = load_criteria(scheme)

    if verbose:
        typer.echo(f"  Loaded {len(criteria)} criteria from JSON for scheme: {scheme}")

    engine = get_engine()

    with Session(engine) as session:
        # Delete existing rows for this scheme to ensure idempotency
        deleted = session.execute(
            delete(CriteriaRecord).where(CriteriaRecord.scheme_name == scheme)
        ).rowcount
        if verbose and deleted:
            typer.echo(f"  Removed {deleted} stale row(s) for scheme: {scheme}")

        records = [CriteriaRecord.from_pydantic(c) for c in criteria]
        session.add_all(records)
        session.commit()

    return len(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    action: Action = typer.Option(
        ...,
        "--action",
        help="'init' to create the database; 'load' to import criteria for a scheme.",
    ),
    scheme: Optional[str] = typer.Option(
        None,
        "--scheme",
        help="Certification scheme to load.  Required when --action=load.",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print progress."),
) -> None:
    """Persist structured criteria to SQLite (Stage 5)."""
    if action == Action.init:
        init_db()
        if verbose:
            typer.echo(f"Database initialised -> {DB_PATH}")

    elif action == Action.load:
        if scheme is None:
            typer.echo("Error: --scheme is required when --action=load.", err=True)
            raise typer.Exit(code=1)

        scheme = scheme.upper()

        # Ensure tables exist before loading
        init_db()

        try:
            n = load_scheme(scheme, verbose=verbose)
        except FileNotFoundError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)

        if verbose:
            typer.echo(f"\nLoad complete: {n} row(s) inserted for scheme: {scheme}")
            typer.echo(f"Database: {DB_PATH}")


if __name__ == "__main__":
    app()
