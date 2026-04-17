"""
Feature engineering — compute per-account signals from usage, support, NPS,
SDK version, and changelog exposure.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config


# ─────────────────────────────────────────────────────────────────────────
#  Usage features
# ─────────────────────────────────────────────────────────────────────────

def _usage_features(usage: pd.DataFrame) -> pd.DataFrame:
    """
    Per-account usage features:
      - api_trend:     % change (avg of last 3 months vs first 3 months)
      - user_trend:    same for active_users
      - content_trend: same for content_entries_created
      - workflow_trend: same for workflows_triggered
      - latest_sdk:    most recent sdk_version string
      - sdk_is_v3:     True if latest SDK is v3.x
    """
    usage = usage.sort_values(["account_id", "month"])

    def _compute(grp: pd.DataFrame) -> pd.Series:
        n = len(grp)
        half = n // 2
        early = grp.iloc[:half]
        late  = grp.iloc[half:]

        def pct(col):
            e = early[col].mean()
            l = late[col].mean()
            if e == 0:
                return 0.0
            return (l - e) / e * 100

        latest_sdk = grp["sdk_version"].iloc[-1]
        return pd.Series({
            "api_trend":       round(pct("api_calls"), 1),
            "user_trend":      round(pct("active_users"), 1),
            "content_trend":   round(pct("content_entries_created"), 1),
            "workflow_trend":  round(pct("workflows_triggered"), 1),
            "latest_sdk":      latest_sdk,
            "sdk_is_v3":       str(latest_sdk).startswith("v3."),
            "avg_api_calls":   round(grp["api_calls"].mean(), 0),
            "avg_active_users": round(grp["active_users"].mean(), 1),
            "latest_api_calls": int(late["api_calls"].iloc[-1]) if len(late) > 0 else 0,
            "latest_active_users": int(late["active_users"].iloc[-1]) if len(late) > 0 else 0,
        })

    return usage.groupby("account_id").apply(_compute, include_groups=False).reset_index()


# ─────────────────────────────────────────────────────────────────────────
#  Support features
# ─────────────────────────────────────────────────────────────────────────

def _support_features(support: pd.DataFrame) -> pd.DataFrame:
    """
    Per-account support features:
      - ticket_count, open_tickets, escalated_tickets
      - p1_count, p2_count
      - recurring_count (tickets mentioning "recurring" / "third time")
      - avg_resolution_hours
      - recent_ticket_count (last 90 days)
    """
    cutoff = pd.Timestamp(config.REFERENCE_DATE) - pd.Timedelta(days=90)

    def _compute(grp: pd.DataFrame) -> pd.Series:
        recent = grp[grp["created_date"] >= cutoff]
        desc_lower = grp["description"].str.lower()
        return pd.Series({
            "ticket_count":        len(grp),
            "open_tickets":        (grp["status"] == "Open").sum(),
            "escalated_tickets":   (grp["status"] == "Escalated").sum(),
            "p1_count":            (grp["priority"] == "P1").sum(),
            "p2_count":            (grp["priority"] == "P2").sum(),
            "recurring_count":     desc_lower.str.contains("recurring|third time", regex=True, na=False).sum(),
            "blocking_count":      desc_lower.str.contains("blocking", na=False).sum(),
            "avg_resolution_hours": round(grp["resolution_time_hours"].mean(), 1) if grp["resolution_time_hours"].notna().any() else None,
            "recent_ticket_count": len(recent),
        })

    return support.groupby("account_id").apply(_compute, include_groups=False).reset_index()


# ─────────────────────────────────────────────────────────────────────────
#  NPS features
# ─────────────────────────────────────────────────────────────────────────

def _nps_features(nps: pd.DataFrame) -> pd.DataFrame:
    """NPS score and verbatim comment per account."""
    return nps.rename(columns={"score": "nps_score", "verbatim_comment": "nps_comment"})


# ─────────────────────────────────────────────────────────────────────────
#  CSM note features
# ─────────────────────────────────────────────────────────────────────────

def _note_features(notes_by_account: Dict[int, str]) -> pd.DataFrame:
    """
    Extract features from reconciled CSM notes:
      - competitor_mentioned: bool
      - negative_keyword_count: int
      - has_csm_note: bool
      - csm_note_text: raw text for LLM
    """
    rows = []
    for aid, text in notes_by_account.items():
        text_lower = text.lower()
        competitor = any(kw in text_lower for kw in config.COMPETITOR_KEYWORDS)
        neg_count = sum(1 for kw in config.NEGATIVE_KEYWORDS if kw in text_lower)
        rows.append({
            "account_id":           aid,
            "competitor_mentioned": competitor,
            "negative_keyword_count": neg_count,
            "has_csm_note":         True,
            "csm_note_text":        text,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["account_id", "competitor_mentioned", "negative_keyword_count",
                 "has_csm_note", "csm_note_text"]
    )


# ─────────────────────────────────────────────────────────────────────────
#  Renewal window filter
# ─────────────────────────────────────────────────────────────────────────

def filter_renewal_window(accounts: pd.DataFrame) -> pd.DataFrame:
    """Keep only accounts renewing within the 90-day window."""
    return accounts[
        (accounts["contract_end_date"] >= config.REFERENCE_DATE)
        & (accounts["contract_end_date"] <= config.RENEWAL_CUTOFF)
    ].copy()


# ─────────────────────────────────────────────────────────────────────────
#  Build master feature table
# ─────────────────────────────────────────────────────────────────────────

def build_features(
    accounts: pd.DataFrame,
    usage: pd.DataFrame,
    support: pd.DataFrame,
    nps: pd.DataFrame,
    notes_by_account: Dict[int, str],
) -> pd.DataFrame:
    """
    Build the full feature table for accounts in the renewal window.
    Returns one row per account with all computed features.
    """
    # Filter to renewal window
    renewal_accounts = filter_renewal_window(accounts)

    # Compute per-source features
    uf = _usage_features(usage)
    sf = _support_features(support)
    nf = _nps_features(nps)
    cf = _note_features(notes_by_account)

    # Merge everything onto renewal accounts
    df = renewal_accounts.merge(uf, on="account_id", how="left")
    df = df.merge(sf, on="account_id", how="left")
    df = df.merge(nf, on="account_id", how="left")
    df = df.merge(cf, on="account_id", how="left")

    # Fill missing values
    numeric_cols = [
        "api_trend", "user_trend", "content_trend", "workflow_trend",
        "ticket_count", "open_tickets", "escalated_tickets",
        "p1_count", "p2_count", "recurring_count", "blocking_count",
        "recent_ticket_count", "negative_keyword_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df["has_csm_note"]         = df["has_csm_note"].fillna(False)
    df["competitor_mentioned"] = df["competitor_mentioned"].fillna(False)
    df["sdk_is_v3"]            = df["sdk_is_v3"].fillna(False)
    df["nps_score"]            = df["nps_score"].fillna(-1)  # -1 = no response
    df["nps_comment"]          = df["nps_comment"].fillna("")
    df["csm_note_text"]        = df["csm_note_text"].fillna("")

    # Compute days to renewal
    df["days_to_renewal"] = df["contract_end_date"].apply(
        lambda d: (d - config.REFERENCE_DATE).days
    )

    return df.reset_index(drop=True)
