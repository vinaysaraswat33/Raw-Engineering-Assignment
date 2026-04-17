"""
Reconciliation — match CSM note blocks to account IDs via explicit IDs and fuzzy name matching.
"""
from __future__ import annotations
from typing import List, Dict, Optional

import pandas as pd
from rapidfuzz import fuzz, process

from .ingest import CSMNoteBlock


def reconcile_csm_notes(
    notes: List[CSMNoteBlock],
    accounts: pd.DataFrame,
    score_cutoff: int = 70,
) -> List[CSMNoteBlock]:
    """
    Attach each CSM note block to an account_id.

    Strategy:
      1. If the block has an explicit_id (regex-parsed), use it directly.
      2. Otherwise, fuzzy-match the mentioned_name against account_name list.
    """
    account_names: Dict[str, int] = dict(
        zip(accounts["account_name"], accounts["account_id"])
    )
    name_list = list(account_names.keys())

    for block in notes:
        # ── Priority 1: Explicit ID ──────────────────────────────────────
        if block.explicit_id is not None:
            if block.explicit_id in accounts["account_id"].values:
                block.matched_account_id = block.explicit_id
                continue

        # ── Priority 2: Fuzzy name match ─────────────────────────────────
        if block.mentioned_name:
            result = process.extractOne(
                block.mentioned_name,
                name_list,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=score_cutoff,
            )
            if result:
                matched_name, score, _ = result
                block.matched_account_id = account_names[matched_name]

    return notes


def build_notes_by_account(notes: List[CSMNoteBlock]) -> Dict[int, str]:
    """
    Group reconciled note blocks by account_id.
    Returns {account_id: combined_text}.
    """
    result: Dict[int, List[str]] = {}
    for block in notes:
        if block.matched_account_id is not None:
            result.setdefault(block.matched_account_id, []).append(block.raw_text)
    return {aid: "\n\n".join(texts) for aid, texts in result.items()}
