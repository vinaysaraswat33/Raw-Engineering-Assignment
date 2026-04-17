"""
Data ingestion — load every source file into usable structures.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict

import pandas as pd

from . import config


# ── CSV loaders ───────────────────────────────────────────────────────────

def load_accounts() -> pd.DataFrame:
    """Load accounts.csv and parse contract_end_date."""
    df = pd.read_csv(config.ACCOUNTS_CSV)
    df["contract_end_date"] = pd.to_datetime(df["contract_end_date"]).dt.date
    return df


def load_usage() -> pd.DataFrame:
    """Load usage_metrics.csv."""
    df = pd.read_csv(config.USAGE_CSV)
    df["month"] = pd.to_datetime(df["month"])
    return df


def load_support() -> pd.DataFrame:
    """Load support_tickets.csv."""
    df = pd.read_csv(config.SUPPORT_CSV)
    df["created_date"] = pd.to_datetime(df["created_date"])
    return df


def load_nps() -> pd.DataFrame:
    """Load nps_responses.csv.  Handle missing verbatim gracefully."""
    df = pd.read_csv(config.NPS_CSV)
    df["verbatim_comment"] = df["verbatim_comment"].fillna("")
    return df


# ── CSM notes parser ─────────────────────────────────────────────────────

@dataclass
class CSMNoteBlock:
    """One note block extracted from csm_notes.txt."""
    raw_text: str
    explicit_id: int | None = None          # parsed from text if present
    mentioned_name: str | None = None       # company name found in text
    date_hint: str | None = None            # date string if found
    matched_account_id: int | None = None   # filled in reconcile step


def parse_csm_notes() -> List[CSMNoteBlock]:
    """Split csm_notes.txt into blocks delimited by '---'."""
    text = config.CSM_NOTES_TXT.read_text(encoding="utf-8")
    raw_blocks = re.split(r"\n---\n", text)
    blocks: List[CSMNoteBlock] = []

    for raw in raw_blocks:
        raw = raw.strip()
        if not raw or raw.startswith("=== CSM"):
            continue

        block = CSMNoteBlock(raw_text=raw)

        # Try to extract explicit account ids:  acct 1001, #1007, (1004), account 1016
        id_match = re.search(
            r"(?:acct|account|#)\s*(\d{4})", raw, re.IGNORECASE
        )
        if id_match:
            block.explicit_id = int(id_match.group(1))

        # Extract a likely company name from the first line
        # Strategy: strip dates, CSM names, account refs, and punctuation,
        # then use whatever remains as candidate for fuzzy matching
        first_line = raw.split("\n")[0]

        # Remove common date patterns
        cleaned = re.sub(
            r"(?:^|\s)(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}(?:,?\s*\d{4})?|"
            r"(?:march|april|may|june|jan|feb)\s+\d{1,2})",
            " ", first_line, flags=re.IGNORECASE,
        )
        # Remove known CSM names
        for csm in ["Carlos Mendez", "Sarah Chen", "David Kim", "Raj Patel",
                     "James Okafor", "Priya Sharma", "Emily Watson",
                     "Anna Kowalski", "James O.", "priya", "Emily W."]:
            cleaned = cleaned.replace(csm, " ")
        # Remove account id refs, separators, and prefixes
        cleaned = re.sub(r"(?:acct|account|#)\s*\d{4}", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[\-\|#()\[\]{}]", " ", cleaned)
        cleaned = re.sub(r"\d{4}", " ", cleaned)  # standalone 4-digit numbers
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—\t")

        if cleaned and len(cleaned) > 2:
            block.mentioned_name = cleaned

        blocks.append(block)

    return blocks


# ── Changelog parser ─────────────────────────────────────────────────────

@dataclass
class ChangelogEntry:
    """Structured changelog entry."""
    version: str
    date: str
    category: str  # e.g. "Deprecations", "Breaking Changes", "Bug Fixes"
    text: str


def parse_changelog() -> List[ChangelogEntry]:
    """Parse changelog.md into structured entries."""
    text = config.CHANGELOG_MD.read_text(encoding="utf-8")
    entries: List[ChangelogEntry] = []
    current_version = ""
    current_date = ""
    current_category = ""

    for line in text.split("\n"):
        line = line.strip()
        # Version header:  ### v4.3.0 — December 15, 2025
        ver_match = re.match(r"###\s+(v[\d.]+(?:\s*—\s*.+)?)", line)
        if ver_match:
            parts = ver_match.group(1).split("—")
            current_version = parts[0].strip()
            current_date = parts[1].strip() if len(parts) > 1 else ""
            continue

        # Category header:  **Deprecations**
        cat_match = re.match(r"\*\*(.+?)\*\*", line)
        if cat_match and not line.startswith("-"):
            current_category = cat_match.group(1)
            continue

        # Bullet point
        if line.startswith("- "):
            clean = re.sub(r"^-\s*[⚠️🔴]*\s*", "", line).strip()
            if clean and current_version:
                entries.append(ChangelogEntry(
                    version=current_version,
                    date=current_date,
                    category=current_category,
                    text=clean,
                ))

    return entries


# ── Convenience: load everything at once ──────────────────────────────────

@dataclass
class RawData:
    """Container for all raw loaded data."""
    accounts: pd.DataFrame = field(default_factory=pd.DataFrame)
    usage: pd.DataFrame = field(default_factory=pd.DataFrame)
    support: pd.DataFrame = field(default_factory=pd.DataFrame)
    nps: pd.DataFrame = field(default_factory=pd.DataFrame)
    csm_notes: List[CSMNoteBlock] = field(default_factory=list)
    changelog: List[ChangelogEntry] = field(default_factory=list)


def load_all() -> RawData:
    """Load every data source into a single container."""
    return RawData(
        accounts=load_accounts(),
        usage=load_usage(),
        support=load_support(),
        nps=load_nps(),
        csm_notes=parse_csm_notes(),
        changelog=parse_changelog(),
    )
