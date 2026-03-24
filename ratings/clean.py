"""
clean.py — Stage 2: Markdown Cleaning & Validation.

Supports a semi-manual review workflow:

1.  Copy raw markdown files to ``data/cleaned/<scheme>/`` for editing.
2.  Validate that a cleaned file conforms to the project's markdown
    convention (frontmatter, heading role tags, block data-type tags, …).
3.  List files that still need a cleaned counterpart.

CLI usage
---------
# List raw files that have no corresponding cleaned version
python -m ratings.clean --list [--scheme BNB]

# Validate a specific cleaned file
python -m ratings.clean --validate data/cleaned/BNB/foo.md

# Copy raw files that have no cleaned counterpart yet
python -m ratings.clean --copy --scheme BNB
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import typer
import yaml

from ratings.config import CLEANED_DIR, RAW_DIR

log = logging.getLogger(__name__)

app = typer.Typer(
    name="clean",
    help="Stage 2 — Markdown cleaning and validation helper.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ROLE_TAGS: frozenset[str] = frozenset(
    ["category", "credit", "criterion", "prerequisite", "general"]
)

VALID_BLOCK_TAGS: frozenset[str] = frozenset(
    ["table", "figure", "formula", "note"]
)

VALID_CONTENT_TYPES: frozenset[str] = frozenset(["general", "credits"])

# Regex patterns
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_ROLE_TAG_RE = re.compile(r"\[([a-z]+)\]")
_BLOCK_TAG_RE = re.compile(r"^>\s*\[([a-z]+)\]\s*(.*)")
_PAGE_MARKER_RE = re.compile(r"<!--\s*Page\s+\d+\s*-->")
_TABLE_ROW_RE = re.compile(r"^\|")
_FIGURE_RE = re.compile(r"!\[.*?\]|figure not extractable", re.IGNORECASE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)

REQUIRED_FRONTMATTER_FIELDS: tuple[str, ...] = ("scheme", "version", "document", "content_type")


# ---------------------------------------------------------------------------
# Auto-preparation: frontmatter extraction and body cleanup
# ---------------------------------------------------------------------------

_BNB_GENERAL_SECTIONS: frozenset[str] = frozenset([
    "herstellung", "nutzung", "entsorgung",
    "relevanz und zielsetzung",
    "beschreibung",
    "methode",
    "direkt in bezug genommene regelwerke",
    "weitere regelwerke",
    "fachinformationen und anwendungshilfen", "fachinformationen", "anwendungshilfen",
    "erforderliche unterlagen",
    "für die bewertung erforderliche unterlagen",
    "hinweise zur nachweisführung", "hinweise",
    "bewertungsmaßstab",
    "ausnahmeregelungen",
    "fußnoten",
])

_BNB_PREREQUISITE_SECTIONS: frozenset[str] = frozenset([
    "voraussetzung", "voraussetzungen",
])


def _bnb_section_tag(name: str) -> str:
    """Return the role tag for a BNB section name."""
    low = name.lower().strip()
    if low in _BNB_PREREQUISITE_SECTIONS or low.startswith("voraussetzung"):
        return "prerequisite"
    return "general"


def _fmt_yaml_scalar(v: str) -> str:
    """Format a string value for safe inline YAML output."""
    if not v:
        return "''"
    needs_quote = any(c in v for c in ":#{}[]|>&*!,?") or v.startswith(("-", "'"))
    if needs_quote:
        escaped = v.replace("'", "''")
        return f"'{escaped}'"
    return v


def _extract_bnb_meta(stem: str, raw: str) -> dict:
    """Extract frontmatter values from a BNB filename stem + raw content.

    Handles two pymupdf4llm output formats:
    - Format A: page header rendered as a markdown table
      (``||**BNB**|...|``, ``|**Hauptkriteriengruppe**|...|``)
    - Format B: page header collapsed into a single bold paragraph
      (``**Bewertungssystem ... Hauptkriteriengruppe Ökologische Qualität ...**``)
    """
    meta: dict = {}

    # Version from filename: BNB_BN2015_xxx → BN2015
    vm = re.search(r"_BN(\d{4})_", stem, re.IGNORECASE)
    if vm:
        meta["version"] = f"BN{vm.group(1)}"

    # Credit ID from filename: BNB_BN2015_111 → BNB_BN 1.1.1
    cm = re.search(r"_BN\d{4}_(\d+)", stem, re.IGNORECASE)
    if cm:
        digits = cm.group(1)
        meta["credit_id"] = f"BNB_BN {'.'.join(digits)}"

    # Credit name from Document Title comment (strip leading N.N.N prefix)
    tm = re.search(r"<!--\s*Document Title:\s*[\d.]*\s*(.+?)\s*-->", raw)
    if tm:
        meta["credit_name"] = tm.group(1).strip()

    # ── Format A (table): specific cell regexes ──────────────────────────────
    # building_type: row ||**Büro- und Verwaltungsgebäude**|**1.1.x**|
    bt_a = re.search(r"\|\|\*\*([^|*]{5,80})\*\*\|\*\*[\d.]+\*\*\|", raw)
    if bt_a:
        meta["building_type"] = bt_a.group(1).strip()

    # category: row |**Hauptkriteriengruppe**|**Ökologische Qualität**||
    cat_a = re.search(r"\|\*\*Hauptkriteriengruppe\*\*\|\*\*([^|*]+?)\*\*\|", raw)
    if cat_a:
        meta["category"] = cat_a.group(1).strip()

    # subcategory: row |**Kriteriengruppe**|**...**||
    sub_a = re.search(r"\|\*\*Kriteriengruppe\*\*\|\*\*([^|*]+?)\*\*\|", raw)
    if sub_a:
        meta["subcategory"] = sub_a.group(1).strip()

    # ── Format B (bold paragraph): fallback when table patterns found nothing ─
    if not meta.get("category"):
        hm = re.search(
            r"Hauptkriteriengruppe\s+(.+?)\s+Kriteriengruppe\s+(.+?)\s+Kriterium\b",
            raw, re.DOTALL,
        )
        if hm:
            meta["category"] = re.sub(r"\s+", " ", hm.group(1)).strip()
            meta["subcategory"] = re.sub(r"\s+", " ", hm.group(2)).strip()

    if not meta.get("building_type"):
        bt_b = re.search(r"Büro-?\s*und\s*Verwaltungsgebäude", raw)
        if bt_b:
            meta["building_type"] = "Büro- und Verwaltungsgebäude"

    return meta


def _is_bnb_page_header(s: str) -> bool:
    """Return True if *s* is a BNB repetitive page-header line to discard."""
    if "Bewertungssystem Nachhaltiges Bauen (BNB)" in s:
        return True
    if re.match(r"^\*\*BNB_BN\s+[\d.]+\*\*\s*$", s):
        return True
    if re.match(r"^\*\*©\s*BMUB", s):
        return True
    if re.match(r"^\*\*[AB]\d+\*\*\s*$", s):   # A1, A2, B1 etc.
        return True
    if re.match(r"^>\s*\*\*(Hauptkriteriengruppe|Kriteriengruppe|Kriterium)\b", s):
        return True
    if "==> picture" in s and "omitted" in s:
        return True
    return False


# ---------------------------------------------------------------------------
# BNB section-label splitting
# ---------------------------------------------------------------------------
# Each entry is (compiled_pattern, replacement_string).  Applied in document
# order to the raw markdown text before the line-by-line pass so that even
# single-paragraph (all-in-one-line) files are split correctly.
#
# Rules:
#  1. Merged spans (two adjacent label parts in one **…** run) — most specific.
#  2. Full single-span labels.
#  3. Split-label openers (first half when the PDF two-column layout broke the
#     label across a line/boundary).
#  4. Split-label closers (second half — strip the bold markup, keep the text).

_SECTION_LABEL_SUBS: list[tuple[re.Pattern, str]] = [
    # ── merged spans ────────────────────────────────────────────────────────
    (re.compile(r"\*\*genommene Regelwerke\s+Weitere Regelwerke\*\*"),
     "\n\n#### Weitere Regelwerke [general]\n\n"),
    (re.compile(r"\*\*und Anwendungshilfen\s+Für die Bewertung\*\*"),
     "\n\n#### Für die Bewertung erforderliche Unterlagen [general]\n\n"),
    (re.compile(r"\*\*und Anwendungshilfen\s+Erforderliche\*\*"),
     "\n\n#### Erforderliche Unterlagen [general]\n\n"),
    (re.compile(r"\*\*Anwendungshilfen\s+Für die Bewertung\*\*"),
     "\n\n#### Für die Bewertung erforderliche Unterlagen [general]\n\n"),
    (re.compile(r"\*\*Anwendungshilfen\s+Erforderliche\*\*"),
     "\n\n#### Erforderliche Unterlagen [general]\n\n"),
    (re.compile(r"\*\*erforderliche Unterlagen\s+Hinweise zur\*\*"),
     "\n\n#### Hinweise zur Nachweisführung [general]\n\n"),
    (re.compile(r"\*\*Unterlagen\s+Hinweise zur\*\*"),
     "\n\n#### Hinweise zur Nachweisführung [general]\n\n"),
    # ── full single-span labels ───────────────────────────────────────────────
    (re.compile(r"\*\*Relevanz und Zielsetzung\*\*"),
     "\n\n#### Relevanz und Zielsetzung [general]\n\n"),
    (re.compile(r"\*\*Beschreibung\*\*"),
     "\n\n#### Beschreibung [general]\n\n"),
    (re.compile(r"\*\*Methode\*\*"),
     "\n\n#### Methode [general]\n\n"),
    (re.compile(r"\*\*Weitere Regelwerke\*\*"),
     "\n\n#### Weitere Regelwerke [general]\n\n"),
    (re.compile(r"\*\*Direkt in Bezug genommene Regelwerke\*\*"),
     "\n\n#### Direkt in Bezug genommene Regelwerke [general]\n\n"),
    (re.compile(r"\*\*Fachinformationen und Anwendungshilfen\*\*"),
     "\n\n#### Fachinformationen und Anwendungshilfen [general]\n\n"),
    (re.compile(r"\*\*Erforderliche Unterlagen\*\*"),
     "\n\n#### Erforderliche Unterlagen [general]\n\n"),
    (re.compile(r"\*\*Für die Bewertung erforderliche Unterlagen\*\*"),
     "\n\n#### Für die Bewertung erforderliche Unterlagen [general]\n\n"),
    (re.compile(r"\*\*Hinweise zur Nachweisführung\*\*"),
     "\n\n#### Hinweise zur Nachweisführung [general]\n\n"),
    # ── split-label openers (first bold span of a split label) ───────────────
    (re.compile(r"\*\*Relevanz und\*\*\s*"),
     "\n\n#### Relevanz und Zielsetzung [general]\n\n"),
    (re.compile(r"\*\*Direkt in Bezug\*\*\s*"),
     "\n\n#### Direkt in Bezug genommene Regelwerke [general]\n\n"),
    (re.compile(r"\*\*Fachinformationen\*\*\s*"),
     "\n\n#### Fachinformationen und Anwendungshilfen [general]\n\n"),
    (re.compile(r"\*\*Erforderliche\*\*\s*"),
     "\n\n#### Erforderliche Unterlagen [general]\n\n"),
    (re.compile(r"\*\*Für die Bewertung\*\*\s*"),
     "\n\n#### Für die Bewertung erforderliche Unterlagen [general]\n\n"),
    (re.compile(r"\*\*Hinweise zur\*\*\s*"),
     "\n\n#### Hinweise zur Nachweisführung [general]\n\n"),
    # ── split-label closers (second bold span — strip the marker) ────────────
    (re.compile(r"\*\*Zielsetzung\*\*\s*"), ""),
    (re.compile(r"\*\*genommene Regelwerke\*\*\s*"), ""),
    (re.compile(r"\*\*und Anwendungshilfen\*\*\s*"), ""),
    (re.compile(r"\*\*und\*\*\s*"), ""),   # stray connector between Fachinformationen items
    (re.compile(r"\*\*Anwendungshilfen\*\*\s*"), ""),
    (re.compile(r"\*\*erforderliche Unterlagen\*\*\s*"), ""),
    (re.compile(r"\*\*Unterlagen\*\*\s*"), ""),
    (re.compile(r"\*\*Nachweisführung\*\*\s*"), ""),
]


def _split_section_labels(text: str) -> str:
    """Replace inline bold BNB section labels with ``#### Heading [general]``.

    Skips lines that already start with ``#`` (headings) or ``|`` (table rows)
    to avoid double-converting headings in complex multi-page documents.
    Collapses any runs of three or more blank lines left by the substitutions.
    """
    result_lines: list[str] = []
    for line in text.split("\n"):
        s = line.lstrip()
        if s.startswith("#") or s.startswith("|") or s.startswith("<!--"):
            result_lines.append(line)
        else:
            for pattern, replacement in _SECTION_LABEL_SUBS:
                line = pattern.sub(replacement, line)
            result_lines.append(line)
    result = "\n".join(result_lines)
    # Collapse runs of 3+ blank lines introduced by the substitutions
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def _is_scoring_row(line: str) -> bool:
    """Return True if *line* is a BNB scoring-table data or note row."""
    return bool(re.search(r"\|\|(?:[ZRG]: \d+|0)\|", line)) or "Zwischenwerte" in line


# ---------------------------------------------------------------------------
# Footnote conversion
# ---------------------------------------------------------------------------

# Regex for inline footnote references: a 1-2 digit number followed by )
# that appears immediately after a non-whitespace, non-digit, non-open-paren
# character.  This matches patterns like ``(SVHC)1)``, ``Regelungen3)``,
# ``\u201eKandidatenliste\u20195)`` but NOT ``(QN 1)`` (space before digit) or
# ``2015)`` (digit before digit).
_FN_REF_RE = re.compile(r"(?<=[^\s\d(])(\d{1,2})\)(?!\d)")

# Match a footnote definition at the start of a stripped line: ``12) text...''
_FN_DEF_RE = re.compile(r"^(\d{1,2})\)\s+(.*)")

# Match a new footnote starting mid-line after a sentence-ending period:
# ``... text. 4) New footnote...''  The match is replaced with a line break
# so that each definition ends up on its own line.  We require the text
# after ``) '' to start with an uppercase letter to avoid matching legal
# citations like ``(REACH Art. 33) gilt...''.
_FN_MID_RE = re.compile(r"(?<=[.!?])\s+(\d{1,2})\)\s+(?=[A-ZÄÖÜ])")


def _convert_footnotes(text: str) -> str:
    """Convert BNB footnote markers to standard Markdown footnote syntax.

    Two transformations:
    - Inline references such as ``(SVHC)1)`` become ``(SVHC)[^1]``.
    - Definition lines starting with ``N) text`` become ``[^N]: text``.

    This makes the cross-references machine-readable: an LLM encountering
    ``[^1]`` in the body can resolve it by finding ``[^1]:`` in the
    ``#### Fußnoten`` section.
    """
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Step 1: convert line-start definitions: ``12) text'' → ``[^12]: text''
        # Also split any embedded sub-definitions within this definition line,
        # e.g. ``3) text. 4) other'' → ``[^3]: text.\n\n[^4]: other''.
        # (_FN_MID_RE is intentionally applied ONLY here to avoid false
        # positives like legal citations ``(REACH Art. 33) gilt...''.)
        dm = _FN_DEF_RE.match(stripped)
        if dm:
            defn = _FN_MID_RE.sub(lambda m: f".\n\n[^{m.group(1)}]: ", dm.group(2))
            result.append(f"[^{dm.group(1)}]: {defn}")
            continue
        # Step 2: convert inline refs: ``word3)'' → ``word[^3]''
        line = _FN_REF_RE.sub(lambda m: f"[^{m.group(1)}]", line)
        result.append(line)
    return "\n".join(result)


_NOTE_PAGE_CLEAN_RE = re.compile(r"^>\s*\[note\]\s*Page\s+\d+\s*$")


def _dedup_page_headings(text: str) -> str:
    """Remove heading lines that are page-continuation repeats.

    In BNB PDFs the section header is reprinted at the top of each page when
    a section spans multiple pages.  After ``_split_section_labels`` converts
    those inline bold labels to proper ``####`` headings, a long section like
    *Methode* ends up with the same ``#### Methode [general]`` heading inserted
    after every page break.

    This function scans the body for headings that appear immediately after a
    ``> [note] Page N`` marker (only blank lines between the marker and the
    heading) and whose text matches the most-recently-seen heading **at the
    same level**.  Matches are removed and replaced with an invisible HTML
    comment as a review hint.

    A higher-level heading *resets* the tracking for all deeper levels so that
    same-named sub-headings that legitimately recur in different `####`
    sections (e.g. *Handhabe bei der Bewertung* under Schwermetalle and again
    under VOC) are treated as fresh occurrences.
    """
    lines = text.splitlines()
    result: list[str] = []
    last_heading: dict[int, str] = {}   # level → heading text (with tag)
    after_page = False                   # True while we've only seen blanks since last page marker

    for line in lines:
        stripped = line.strip()

        if _NOTE_PAGE_CLEAN_RE.match(stripped):
            after_page = True
            result.append(line)
            continue

        if not stripped:
            # Blank lines never reset the after-page window
            result.append(line)
            continue

        hm = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if hm:
            level = len(hm.group(1))
            heading_text = hm.group(2).strip()
            # Reset tracking for all deeper levels when a parent heading fires
            for k in list(last_heading.keys()):
                if k > level:
                    del last_heading[k]
            if after_page and last_heading.get(level) == heading_text:
                # Page-continuation duplicate — silently drop it
                after_page = False
                continue
            last_heading[level] = heading_text

        # Only reset the page-continuation window for *substantial* content.
        # Lone bullet markers (``- `` with nothing after) are PDF layout
        # artefacts that should not break the dedup look-ahead.
        if stripped and not re.match(r'^[-*+]\s*$', stripped):
            after_page = False
        result.append(line)

    return "\n".join(result)


def _prepare_bnb_body(raw: str, meta: dict) -> str:
    """Transform a raw BNB markdown body into a clean, role-tagged body."""
    category = meta.get("category", "")
    subcategory = meta.get("subcategory", "")
    credit_name = meta.get("credit_name", "")

    # Pre-process: split inline bold section labels into proper headings.
    # This handles the "all sections on one line" case produced by some PDFs.
    raw = _split_section_labels(raw)

    lines = raw.splitlines()
    out: list[str] = []

    # Prepend the heading hierarchy
    if category:
        out += [f"# {category} [category]", ""]
    if subcategory:
        out += [f"## {subcategory} [category]", ""]
    if credit_name:
        out += [f"### {credit_name} [criterion]", ""]

    prev_blank = True
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Drop Document Title comment
        if stripped.startswith("<!-- Document Title:"):
            i += 1
            continue

        # Page marker — convert to > [note] Page N, then skip following BNB header lines
        if re.match(r"^<!--\s*Page\s+(\d+)\s*-->", stripped):
            pm = re.match(r"^<!--\s*Page\s+(\d+)\s*-->", stripped)
            out += ["", f"> [note] Page {pm.group(1)}", ""]
            prev_blank = True
            i += 1
            # Skip BNB page-header block immediately after the page marker
            while i < len(lines) and (
                not lines[i].strip() or _is_bnb_page_header(lines[i].strip())
            ):
                i += 1
            continue

        # Drop any remaining page-header lines (e.g. from page 1 before first marker)
        if _is_bnb_page_header(stripped):
            i += 1
            continue

        # Drop single-letter heading artefacts from two-column PDF extraction
        # e.g. ``## G'' which are misidentified bullet markers.
        if re.match(r"^#{1,4}\s+[A-Z]\s*$", stripped):
            i += 1
            continue

        # Convert: ## **SectionName** → #### or ##### SectionName [tag]
        #
        # BNB PDFs encode ALL bold headings at the same markdown level (## **…**).
        # We need to distinguish:
        # - Known *section labels* (Methode, Beschreibung, etc.) which are direct
        #   children of the criterion → output as ####  (level 4).
        # - *Structural sub-headings* (Stoffe und Gemische, 1. Gefährliche Stoffe,
        #   etc.) which live *inside* a section → output as #####  (level 5).
        # This prevents sub-headings from clobbering the dedup tracker that relies
        # on ####-level identities to strip page-continuation heading repeats.
        #
        # Also handles italic-bold variants: ## _**Name**_
        hm = re.match(r"^#{1,4}\s+_?\*\*(.+?)\*\*_?\s*$", stripped)
        if hm:
            name = hm.group(1).strip()
            # Skip lone footnote-anchor headings like ``## **4)**'' that appear
            # in some PDFs as misplaced superscript markers.
            if re.match(r"^\d+\)$", name):
                i += 1
                continue
            tag = _bnb_section_tag(name)
            low = name.lower()
            is_section_label = (
                low in _BNB_GENERAL_SECTIONS or low in _BNB_PREREQUISITE_SECTIONS
            )
            level_mark = "####" if is_section_label else "#####"
            out.append(f"{level_mark} {name} [{tag}]")
            prev_blank = False
            i += 1
            continue

        # Convert: ## **SectionLabel** body_continues (run-on heading+text from PDF)
        # e.g. ``## **Methode** c)  Ausschluss bestimmter …''
        hmx = re.match(r"^#{1,4}\s+\*\*([^*]{2,40})\*\*\s+(\S.+)$", stripped)
        if hmx:
            label = hmx.group(1).strip()
            rest  = hmx.group(2)
            low   = label.lower()
            if low in _BNB_GENERAL_SECTIONS or low in _BNB_PREREQUISITE_SECTIONS:
                tag = _bnb_section_tag(label)
                out += [f"#### {label} [{tag}]", "", rest]
                prev_blank = False
                i += 1
                continue

        # Convert: standalone bold line matching a known section name
        sm = re.match(r"^\*\*([^*]{2,60})\*\*\s*$", stripped)
        if sm:
            name = sm.group(1).strip()
            low = name.lower()
            if low in _BNB_GENERAL_SECTIONS or low in _BNB_PREREQUISITE_SECTIONS:
                tag = _bnb_section_tag(name)
                out.append(f"#### {name} [{tag}]")
                prev_blank = False
                i += 1
                continue

        # Convert: inline section label at line start: **Label** rest of text
        ilm = re.match(r"^\*\*([^*]{2,40})\*\*\s+(\S.+)$", stripped)
        if ilm:
            label = ilm.group(1).strip()
            rest = ilm.group(2)
            if label.lower() in _BNB_GENERAL_SECTIONS or label.lower() in _BNB_PREREQUISITE_SECTIONS:
                tag = _bnb_section_tag(label)
                out += [f"#### {label} [{tag}]", "", rest]
                prev_blank = False
                i += 1
                continue

        # Fix bullet artifact: "- G text" → "- text"
        line = re.sub(r"^(\s*)-\s+G\s+", r"\1- ", line)

        # Normalise PDF bullet character → Markdown dash
        line = re.sub(r"^(\s*)\u2022\s+", r"\1- ", line)

        # Convert indented letter-list items (a), b), c) …) to proper
        # Markdown list items so nested list structure is preserved.
        # Requires 2+ spaces of indentation to avoid false positives.
        line = re.sub(r"^(\s{2,})([a-z])\)\s+", r"\1- \2) ", line)

        # Table handling: classify tables as BNB header (strip), scoring (keep
        # cleaned), or unknown (keep with placeholder caption).
        if re.match(r"^\|", stripped):
            # Collect all consecutive table lines
            table_lines: list[str] = []
            while i < len(lines) and re.match(r"^\|", lines[i].strip()):
                table_lines.append(lines[i])
                i += 1
            combined = "\n".join(table_lines)
            scoring = [l for l in table_lines if _is_scoring_row(l)]
            is_bnb_meta = (
                "Hauptkriteriengruppe" in combined
                or "Bewertungssystem Nachhaltiges Bauen" in combined
                or bool(scoring and "Anforderungsniveau" in combined)
            )
            if not is_bnb_meta:
                # Regular content table — keep with placeholder caption
                last_nonempty = next((l for l in reversed(out) if l.strip()), "")
                if not re.match(r"^>\s*\[table\]", last_nonempty):
                    out.append("> [table] Bewertungsmaßstab (Beschriftung prüfen)")
                out.extend(table_lines)
            elif scoring:
                # BNB scoring table with repetitive metadata header rows —
                # keep only the scoring rows and their Anforderungsniveau header.
                last_nonempty = next((l for l in reversed(out) if l.strip()), "")
                if not re.match(r"^>\s*\[table\]", last_nonempty):
                    out += ["", "> [table] Bewertungsmaßstab", ""]
                sep = next(
                    (l for l in table_lines if re.match(r"^\|---", l.strip())),
                    "|---|---|---|",
                )
                out.append(sep)
                anf = [l for l in table_lines if "Anforderungsniveau" in l]
                out.extend(anf)
                out.extend(scoring)
            # else: pure BNB page-header table → strip entirely (no output)
            prev_blank = False
            continue

        # Blank-line deduplication
        if not stripped:
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(line)
            prev_blank = False

        i += 1

    while out and not out[-1].strip():
        out.pop()

    body = "\n".join(out)
    body = _convert_footnotes(body)
    body = _dedup_page_headings(body)
    return body + "\n"


def _auto_prepare(raw_path: Path, scheme: str) -> tuple[str, str]:
    """Return ``(frontmatter_str, body_str)`` ready for human review.

    Extracts as much structured information as possible from the filename and
    raw content, cleans up common PDF extraction artefacts, and adds role tags
    where deterministic.  The result still requires manual review but should be
    largely complete.
    """
    raw = raw_path.read_text(encoding="utf-8")
    scheme_up = scheme.upper()

    meta = _extract_bnb_meta(raw_path.stem, raw) if scheme_up == "BNB" else {}

    fields = {
        "scheme": scheme_up,
        "version": meta.get("version", ""),
        "document": raw_path.stem,
        "content_type": "credits",
        "credit_id": meta.get("credit_id", ""),
        "credit_name": meta.get("credit_name", ""),
        "category": meta.get("category", ""),
        "building_type": meta.get("building_type", ""),
    }

    fm_lines = ["---"]
    for k, v in fields.items():
        fm_lines.append(f"{k}: {_fmt_yaml_scalar(str(v))}")
    fm_lines.append("---")
    fm_str = "\n".join(fm_lines) + "\n"

    body = _prepare_bnb_body(raw, meta) if scheme_up == "BNB" else raw

    return fm_str, body


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str) -> Optional[dict]:
    """Parse the YAML frontmatter block from *text*.

    Returns the parsed dict, or *None* if no frontmatter is found.
    The frontmatter must be a ``---`` … ``---`` block at the very start of the
    file (optionally preceded by whitespace-only lines).
    """
    stripped = text.lstrip("\n\r ")
    m = _FRONTMATTER_RE.match(stripped)
    if m is None:
        return None
    try:
        data = yaml.safe_load(m.group(1))
        return data if isinstance(data, dict) else None
    except yaml.YAMLError:
        return None


def validate_file(path: Path) -> list[str]:
    """Validate *path* against the project's markdown convention.

    Returns a (possibly empty) list of human-readable violation strings.
    An empty list means the file is valid.
    """
    text = path.read_text(encoding="utf-8")
    violations: list[str] = []

    # ------------------------------------------------------------------
    # Rule 1 & 2: frontmatter
    # ------------------------------------------------------------------
    fm = parse_frontmatter(text)
    if fm is None:
        violations.append("Missing or unparseable YAML frontmatter block (--- ... ---)")
        # Without frontmatter, skip field-level checks but continue structural checks
    else:
        for field in REQUIRED_FRONTMATTER_FIELDS:
            if not fm.get(field):
                violations.append(f"Frontmatter field '{field}' is missing or empty")
        ct = fm.get("content_type", "")
        if ct and ct not in VALID_CONTENT_TYPES:
            violations.append(
                f"Frontmatter 'content_type' must be one of {sorted(VALID_CONTENT_TYPES)}, got {ct!r}"
            )
        # subcategory is no longer a frontmatter field — hierarchy lives in headings
        if fm.get("subcategory"):
            violations.append(
                "Frontmatter field 'subcategory' is deprecated — encode the hierarchy as "
                "nested [category] headings instead and remove this field"
            )

    # Strip frontmatter for body analysis
    stripped = text.lstrip("\n\r ")
    m = _FRONTMATTER_RE.match(stripped)
    body = stripped[m.end():] if m else text

    lines = body.splitlines()

    # ------------------------------------------------------------------
    # Rule 3 & 4: headings and orphaned content
    # ------------------------------------------------------------------
    first_heading_seen = False
    prev_level = 0
    prev_role = ""
    prev_line_block_tag: Optional[str] = None   # tag from the most recent > [tag] line
    prev_is_table_caption = False
    prev_is_figure_caption = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()

        # Skip HTML comments (page markers, title markers)
        if stripped_line.startswith("<!--"):
            i += 1
            prev_line_block_tag = None
            prev_is_table_caption = False
            prev_is_figure_caption = False
            continue

        # Check for blockquote data-type tag line
        bm = _BLOCK_TAG_RE.match(stripped_line)
        if bm:
            tag = bm.group(1)
            if tag not in VALID_BLOCK_TAGS:
                violations.append(
                    f"Line {i + 1}: unknown block tag [{ tag}] "
                    f"(valid: {sorted(VALID_BLOCK_TAGS)})"
                )
            prev_line_block_tag = tag
            prev_is_table_caption = (tag == "table")
            prev_is_figure_caption = (tag == "figure")
            i += 1
            continue

        # Warn on legacy <!-- Page N --> syntax
        if re.match(r"^<!--\s*Page\s+\d+\s*-->", stripped_line):
            violations.append(
                f"Line {i + 1}: legacy HTML page marker — convert to '> [note] Page N' format"
            )
            i += 1
            continue

        # Check heading
        hm = _HEADING_RE.match(line)
        if hm:
            level = len(hm.group(1))
            heading_text = hm.group(2).strip()

            # Rule 3: exactly one valid role tag
            tags_found = _ROLE_TAG_RE.findall(heading_text)
            role_tags = [t for t in tags_found if t in VALID_ROLE_TAGS]
            if len(role_tags) == 0:
                violations.append(
                    f"Line {i + 1}: heading has no role tag — "
                    f"add one of {sorted(VALID_ROLE_TAGS)}: {heading_text!r}"
                )
            elif len(role_tags) > 1:
                violations.append(
                    f"Line {i + 1}: heading has multiple role tags {role_tags!r}: {heading_text!r}"
                )

            # Rule 7: no heading level skip (nested [category] under [category] is always valid)
            this_role = role_tags[0] if len(role_tags) == 1 else ""
            if first_heading_seen and level > prev_level + 1:
                # Allow any depth of nested [category] headings
                if not (this_role == "category" and prev_role == "category"):
                    violations.append(
                        f"Line {i + 1}: heading level skipped — "
                        f"H{prev_level} -> H{level}: {heading_text!r}"
                    )

            first_heading_seen = True
            prev_level = level
            prev_role = this_role
            prev_line_block_tag = None
            prev_is_table_caption = False
            prev_is_figure_caption = False
            i += 1
            continue

        # Rule 4: no orphaned content before the first heading
        if not first_heading_seen and stripped_line:
            violations.append(
                f"Line {i + 1}: non-whitespace content appears before the first heading"
            )
            i += 1
            continue

        # Rule 5: tables must be preceded by a > [table] caption
        if _TABLE_ROW_RE.match(stripped_line):
            if not prev_is_table_caption:
                violations.append(
                    f"Line {i + 1}: markdown table not preceded by '> [table] <caption>'"
                )
            # Consume all consecutive table lines
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i].strip()):
                i += 1
            prev_line_block_tag = None
            prev_is_table_caption = False
            prev_is_figure_caption = False
            continue

        # Rule 6: figures must be preceded by a > [figure] caption
        if _FIGURE_RE.search(stripped_line):
            if not prev_is_figure_caption:
                violations.append(
                    f"Line {i + 1}: figure reference not preceded by '> [figure] <caption>'"
                )
            prev_line_block_tag = None
            prev_is_figure_caption = False
            prev_is_table_caption = False
            i += 1
            continue

        # Any other non-blank line resets the block tag context
        if stripped_line:
            prev_line_block_tag = None
            prev_is_table_caption = False
            prev_is_figure_caption = False

        i += 1

    return violations


def list_pending(scheme: Optional[str] = None) -> list[tuple[Path, Optional[Path]]]:
    """Return raw/cleaned path pairs where the cleaned file is absent.

    Parameters
    ----------
    scheme:
        If given, restrict to one scheme sub-directory.  If *None*, all
        sub-directories under ``data/raw/`` are searched.

    Returns
    -------
    list[tuple[Path, Path | None]]
        Each tuple is ``(raw_path, cleaned_path_or_None)``.  A ``None``
        cleaned path means the cleaned counterpart has not been created yet.
    """
    if scheme is not None:
        search_dirs = [RAW_DIR / scheme.upper()]
    else:
        search_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]

    pending: list[tuple[Path, Optional[Path]]] = []
    for raw_scheme_dir in sorted(search_dirs):
        scheme_name = raw_scheme_dir.name
        for raw_path in sorted(raw_scheme_dir.glob("*.md")):
            cleaned_path = CLEANED_DIR / scheme_name / raw_path.name
            if not cleaned_path.exists():
                pending.append((raw_path, None))
            # Files that do exist are not returned (they are not pending)

    return pending


def copy_to_cleaned(scheme: str) -> list[Path]:
    """Copy raw markdown files to ``data/cleaned/<scheme>/`` for editing.

    Only files that do not yet have a cleaned counterpart are copied.
    Existing cleaned files are never overwritten.

    Parameters
    ----------
    scheme:
        Certification scheme identifier (e.g. ``"BNB"``).

    Returns
    -------
    list[Path]
        Paths of the newly copied cleaned files.
    """
    scheme = scheme.upper()
    raw_dir = RAW_DIR / scheme
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw directory not found: {raw_dir}\n"
            f"Run Stage 1 first:  python -m ratings.ingest --scheme {scheme}"
        )

    out_dir = CLEANED_DIR / scheme
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for raw_path in sorted(raw_dir.glob("*.md")):
        dest = out_dir / raw_path.name
        if dest.exists():
            log.info("Skipping (already exists): %s", dest.relative_to(CLEANED_DIR.parent))
            continue
        fm_str, body = _auto_prepare(raw_path, scheme)
        dest.write_text(fm_str + "\n" + body, encoding="utf-8")
        copied.append(dest)
        log.info("Copied: %s", dest.relative_to(CLEANED_DIR.parent))

    return copied


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    list_pending_flag: bool = typer.Option(
        False,
        "--list",
        help="List raw files that do not yet have a cleaned counterpart.",
    ),
    validate: Optional[Path] = typer.Option(
        None,
        "--validate",
        help="Validate a cleaned markdown file against the convention.",
    ),
    copy: bool = typer.Option(
        False,
        "--copy",
        help="Copy raw files without a cleaned version to data/cleaned/<scheme>/.",
    ),
    scheme: Optional[str] = typer.Option(
        None,
        "--scheme",
        help="Certification scheme identifier (e.g. BNB). Required for --copy.",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print progress."),
) -> None:
    """Markdown cleaning helper — list, validate, or copy files (Stage 2)."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )

    modes = sum([list_pending_flag, validate is not None, copy])
    if modes == 0:
        typer.echo(
            "Error: provide one of --list, --validate <file>, or --copy --scheme <scheme>.",
            err=True,
        )
        raise typer.Exit(code=1)
    if modes > 1:
        typer.echo("Error: provide only one of --list, --validate, or --copy.", err=True)
        raise typer.Exit(code=1)

    # ---- --list ----
    if list_pending_flag:
        pending = list_pending(scheme)
        if not pending:
            typer.echo("All raw files have corresponding cleaned versions.")
        else:
            typer.echo(f"{len(pending)} file(s) awaiting cleaning:\n")
            for raw_path, _ in pending:
                typer.echo(f"  {raw_path}")
        return

    # ---- --validate ----
    if validate is not None:
        vpath = Path(validate)
        if not vpath.exists():
            typer.echo(f"Error: file not found: {vpath}", err=True)
            raise typer.Exit(code=1)
        violations = validate_file(vpath)
        if not violations:
            typer.echo(f"OK — {vpath} passes all validation rules.")
        else:
            typer.echo(f"FAIL — {len(violations)} violation(s) in {vpath}:\n")
            for v in violations:
                typer.echo(f"  - {v}")
            raise typer.Exit(code=2)
        return

    # ---- --copy ----
    if copy:
        if not scheme:
            typer.echo("Error: --copy requires --scheme.", err=True)
            raise typer.Exit(code=1)
        try:
            copied = copy_to_cleaned(scheme)
        except FileNotFoundError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        if copied:
            typer.echo(f"Copied {len(copied)} file(s) to {CLEANED_DIR / scheme.upper()}")
            typer.echo("")
            typer.echo("=" * 72)
            typer.echo("  !! MANUAL STEP REQUIRED — do NOT continue until this is done !!")
            typer.echo("")
            typer.echo(f"  1. Open each file in data/cleaned/{scheme.upper()}/")
            typer.echo("     - Fill in ALL frontmatter fields (version, credit_id,")
            typer.echo("       credit_name, category, subcategory).")
            typer.echo("     - Add a role tag to every heading, e.g. [category],")
            typer.echo("       [criterion], [general], [prerequisite].")
            typer.echo("     - Precede every markdown table with a caption line:")
            typer.echo("       > [table] <caption text>")
            typer.echo("")
            typer.echo("  2. Validate each file after editing:")
            typer.echo(f"       python -m ratings.clean --validate data/cleaned/{scheme.upper()}/<file>.md")
            typer.echo("")
            typer.echo("  3. Only once ALL files pass --validate, run Stage 3:")
            typer.echo(f"       python -m ratings.chunk --scheme {scheme.upper()}")
            typer.echo("")
            typer.echo("  4. Then run Stage 4 (LLM extraction):")
            typer.echo(f"       python -m ratings.extract --scheme {scheme.upper()}")
            typer.echo("=" * 72)
        else:
            typer.echo("Nothing to copy — all raw files already have cleaned counterparts.")
        return


if __name__ == "__main__":
    app()
