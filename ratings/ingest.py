"""
ingest.py — Stage 1: PDF Ingestion.

Extracts text from PDF manuals using pymupdf4llm, preserving document
structure (headings, tables, lists) as Markdown.  Output is written to
``data/raw/{scheme}/{document_stem}.md``.

CLI usage
---------
# Single PDF
python -m ratings.ingest --pdf path/to/manual.pdf --scheme BREEAM

# All PDFs in a directory
python -m ratings.ingest --pdf-dir path/to/pdfs/ --scheme LEED

# Process inbox folder: picks up all PDFs from data/inbox/{scheme}/
# and moves each to data/done/{scheme}/ after successful extraction.
python -m ratings.ingest --inbox --scheme BNB
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer
import pymupdf4llm

from ratings.config import DONE_DIR, INBOX_DIR, RAW_DIR

app = typer.Typer(
    name="ingest",
    help="Stage 1 — Extract text from PDF manuals and save as Markdown.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def extract_pdf_to_markdown(pdf_path: Path) -> str:
    """Extract text and structure from *pdf_path* and return Markdown.

    Page boundaries are marked with ``<!-- Page N -->`` HTML comments so that
    downstream stages can recover page numbers.  A
    ``<!-- Document Title: ... -->`` comment is embedded on the first line
    using the title stored in the PDF's internal metadata.

    Parameters
    ----------
    pdf_path:
        Absolute or relative path to the source PDF file.

    Returns
    -------
    str
        Full document content as Markdown with embedded page and title markers.
    """
    page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    # Title comes from the PDF metadata (same dict for every page chunk)
    first_meta: dict = page_chunks[0].get("metadata", {}) if page_chunks else {}
    pdf_title: str = (first_meta.get("title") or "").strip() or pdf_path.stem

    parts: list[str] = [f"<!-- Document Title: {pdf_title} -->"]
    for item in page_chunks:
        # page_number is 1-indexed in pymupdf4llm's metadata
        page_num: int = item.get("metadata", {}).get("page_number", 1)
        parts.append(f"<!-- Page {page_num} -->\n\n{item['text']}")

    return "\n\n".join(parts)


def ingest_pdf(pdf_path: Path, scheme: str, *, verbose: bool = True) -> Path:
    """Ingest a single PDF and write its Markdown to ``data/raw/{scheme}/``.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    scheme:
        Certification scheme identifier (e.g. ``"BREEAM"``).  Used as the
        sub-directory name under ``data/raw/``.
    verbose:
        If *True*, print progress messages to stdout.

    Returns
    -------
    Path
        Path to the written ``.md`` file.

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_dir = RAW_DIR / scheme
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{pdf_path.stem}.md"

    if verbose:
        typer.echo(f"  Extracting: {pdf_path.name} -> {out_path.relative_to(RAW_DIR.parent.parent)}")

    md_text = extract_pdf_to_markdown(pdf_path)
    out_path.write_text(md_text, encoding="utf-8")

    if verbose:
        word_count = len(md_text.split())
        typer.echo(f"  Done — {word_count:,} words written to {out_path.name}")

    return out_path


def ingest_directory(pdf_dir: Path, scheme: str, *, verbose: bool = True) -> list[Path]:
    """Ingest all ``*.pdf`` files found in *pdf_dir*.

    Parameters
    ----------
    pdf_dir:
        Directory containing PDF files to process.
    scheme:
        Certification scheme identifier.
    verbose:
        If *True*, print progress messages.

    Returns
    -------
    list[Path]
        Paths of all written ``.md`` files.
    """
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        typer.echo(f"  Warning: no PDF files found in {pdf_dir}", err=True)
        return []

    if verbose:
        typer.echo(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}")

    results: list[Path] = []
    for pdf_path in pdf_files:
        results.append(ingest_pdf(pdf_path, scheme, verbose=verbose))

    return results


def ingest_inbox(scheme: str, *, verbose: bool = True) -> list[Path]:
    """Process all PDFs waiting in ``data/inbox/{scheme}/``.

    Each PDF is extracted to ``data/raw/{scheme}/`` as usual.  After a
    successful extraction the source PDF is moved to ``data/done/{scheme}/``
    so it is never processed twice.  Failed files are left in the inbox.

    Parameters
    ----------
    scheme:
        Certification scheme identifier (e.g. ``"BNB"``).  Determines both
        the inbox sub-folder to read from and the scheme metadata tag.
    verbose:
        If *True*, print progress messages.

    Returns
    -------
    list[Path]
        Paths of all successfully written ``.md`` files.
    """
    inbox = INBOX_DIR / scheme
    done = DONE_DIR / scheme

    if not inbox.exists():
        typer.echo(
            f"Inbox folder not found: {inbox}\n"
            f"Create it and drop PDFs there, then re-run.",
            err=True,
        )
        return []

    pdf_files = sorted(inbox.glob("*.pdf"))
    if not pdf_files:
        typer.echo(f"  No PDFs found in inbox: {inbox}")
        return []

    done.mkdir(parents=True, exist_ok=True)

    if verbose:
        typer.echo(f"Found {len(pdf_files)} PDF(s) in inbox for scheme: {scheme}")

    results: list[Path] = []
    for pdf_path in pdf_files:
        try:
            md_path = ingest_pdf(pdf_path, scheme, verbose=verbose)
            results.append(md_path)
            dest = done / pdf_path.name
            shutil.move(str(pdf_path), dest)
            if verbose:
                typer.echo(f"  Moved -> done/{scheme}/{pdf_path.name}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(
                f"  Error processing {pdf_path.name}: {exc} — left in inbox.",
                err=True,
            )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    pdf: Optional[Path] = typer.Option(
        None,
        "--pdf",
        help="Path to a single PDF file to ingest.",
        exists=False,  # we validate manually to give friendlier error messages
    ),
    pdf_dir: Optional[Path] = typer.Option(
        None,
        "--pdf-dir",
        help="Path to a directory containing PDF files to ingest.",
    ),
    inbox: bool = typer.Option(
        False,
        "--inbox/--no-inbox",
        help=(
            "Process all PDFs in data/inbox/{scheme}/ and move each to "
            "data/done/{scheme}/ after successful extraction."
        ),
    ),
    scheme: str = typer.Option(
        ...,
        "--scheme",
        help="Certification scheme identifier, e.g. BREEAM, LEED, DGNB, BNB.",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print progress."),
) -> None:
    """Extract text from PDF manuals and save as Markdown (Stage 1)."""
    modes = sum([pdf is not None, pdf_dir is not None, inbox])
    if modes == 0:
        typer.echo("Error: provide one of --pdf, --pdf-dir, or --inbox.", err=True)
        raise typer.Exit(code=1)
    if modes > 1:
        typer.echo("Error: provide only one of --pdf, --pdf-dir, or --inbox.", err=True)
        raise typer.Exit(code=1)

    scheme = scheme.upper()

    if pdf is not None:
        try:
            ingest_pdf(Path(pdf), scheme, verbose=verbose)
        except FileNotFoundError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
    elif pdf_dir is not None:
        pdf_dir_path = Path(pdf_dir)
        if not pdf_dir_path.is_dir():
            typer.echo(f"Error: {pdf_dir_path} is not a directory.", err=True)
            raise typer.Exit(code=1)
        ingest_directory(pdf_dir_path, scheme, verbose=verbose)
    else:
        ingest_inbox(scheme, verbose=verbose)
        if verbose:
            typer.echo(f"  Inbox: {INBOX_DIR / scheme}")
            typer.echo(f"  Done:  {DONE_DIR / scheme}")
        return  # inbox prints its own summary

    if verbose:
        typer.echo(f"\nIngestion complete for scheme: {scheme}")
        typer.echo(f"Output directory: {RAW_DIR / scheme}")


if __name__ == "__main__":
    app()
