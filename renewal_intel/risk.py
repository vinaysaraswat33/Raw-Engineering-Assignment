"""
Risk scoring — weighted composite score with structured factor breakdown.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config


def _score_usage_decline(row: pd.Series) -> Tuple[float, str]:
    """Score 0-1 based on API call and content trend decline."""
    api = row.get("api_trend", 0)
    content = row.get("content_trend", 0)
    avg_decline = (min(api, 0) + min(content, 0)) / 2  # negative values
    # Map: 0% → 0, -50% or worse → 1
    score = min(abs(avg_decline) / 50, 1.0)
    if score > 0.3:
        return score, f"Usage declined {abs(avg_decline):.0f}% on average (API & content)"
    return score, ""


def _score_support_burden(row: pd.Series) -> Tuple[float, str]:
    """Score based on open tickets, P1s, escalations, recurring issues."""
    open_t = row.get("open_tickets", 0)
    p1     = row.get("p1_count", 0)
    esc    = row.get("escalated_tickets", 0)
    rec    = row.get("recurring_count", 0)
    block  = row.get("blocking_count", 0)

    raw = (open_t * 3 + p1 * 4 + esc * 2 + rec * 2 + block * 1.5)
    score = min(raw / 30, 1.0)  # cap at 30 raw points

    factors = []
    if p1 > 0:
        factors.append(f"{int(p1)} P1 tickets")
    if open_t > 0:
        factors.append(f"{int(open_t)} open tickets")
    if esc > 0:
        factors.append(f"{int(esc)} escalated")
    if rec > 0:
        factors.append(f"{int(rec)} recurring issues")

    return score, "; ".join(factors)


def _score_nps(row: pd.Series) -> Tuple[float, str]:
    """Score based on NPS (detractor < 7, passive 7-8, promoter 9-10)."""
    nps = row.get("nps_score", -1)
    if nps < 0:  # no response — mild risk signal
        return 0.3, "No NPS survey response"
    if nps <= 4:
        return 1.0, f"NPS score {nps} — strong detractor"
    if nps <= 6:
        return 0.6, f"NPS score {nps} — detractor"
    if nps <= 8:
        return 0.1, f"NPS score {nps} — passive"
    return 0.0, ""  # promoter


def _score_csm_sentiment(row: pd.Series) -> Tuple[float, str]:
    """Score based on CSM note keywords (competitors, negative sentiment)."""
    competitor = row.get("competitor_mentioned", False)
    neg_count  = row.get("negative_keyword_count", 0)

    score = 0.0
    factors = []
    if competitor:
        score += 0.5
        factors.append("Competitor mentioned in CSM notes")
    score += min(neg_count * 0.15, 0.5)
    if neg_count > 0:
        factors.append(f"{int(neg_count)} negative sentiment signals in notes")

    return min(score, 1.0), "; ".join(factors)


def _score_sdk_deprecated(row: pd.Series) -> Tuple[float, str]:
    """Score for SDK v3.x customers facing deprecation deadline."""
    if row.get("sdk_is_v3", False):
        days_left = (config.SDK_V3_SUNSET_DATE - config.REFERENCE_DATE).days
        if days_left <= 14:
            return 1.0, f"On deprecated SDK v3.x — sunset in {days_left} days!"
        elif days_left <= 30:
            return 0.8, f"On deprecated SDK v3.x — sunset in {days_left} days"
        return 0.6, "On deprecated SDK v3.x — migration required"
    return 0.0, ""


def _score_engagement_decay(row: pd.Series) -> Tuple[float, str]:
    """Score based on active user trend decline."""
    user_trend = row.get("user_trend", 0)
    if user_trend < -30:
        return 1.0, f"Active users dropped {abs(user_trend):.0f}%"
    if user_trend < -15:
        return 0.6, f"Active users declined {abs(user_trend):.0f}%"
    if user_trend < 0:
        return 0.2, f"Slight active user decline ({abs(user_trend):.0f}%)"
    return 0.0, ""


def _score_conflicting_signals(row: pd.Series) -> Tuple[float, str]:
    """
    Detect conflicting signals — the 'silent churn' pattern.
    High NPS but declining usage = people like us but are leaving.
    """
    nps = row.get("nps_score", -1)
    api_trend = row.get("api_trend", 0)
    user_trend = row.get("user_trend", 0)

    if nps >= 7 and (api_trend < -25 or user_trend < -25):
        return 1.0, f"SILENT CHURN — NPS {nps} but usage declined {abs(api_trend):.0f}%"
    if nps >= 7 and (api_trend < -10 or user_trend < -10):
        return 0.4, f"Mixed signals — decent NPS ({nps}) but declining usage"
    return 0.0, ""


def _score_contract_urgency(row: pd.Series) -> Tuple[float, str]:
    """Higher urgency for accounts renewing sooner."""
    days = row.get("days_to_renewal", 90)
    if days <= 14:
        return 1.0, f"Renewal in {days} days — immediate attention needed"
    if days <= 30:
        return 0.7, f"Renewal in {days} days"
    if days <= 60:
        return 0.4, f"Renewal in {days} days"
    return 0.1, ""


# ─────────────────────────────────────────────────────────────────────────
#  Main scoring function
# ─────────────────────────────────────────────────────────────────────────

def score_accounts(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute risk score (0-100) and tier for each account.
    Returns the DataFrame augmented with risk_score, risk_tier, risk_factors.
    """
    results = []

    for _, row in features.iterrows():
        scores = {}
        factor_list: List[str] = []

        # Compute each sub-score
        scorers = {
            "usage_decline":       _score_usage_decline,
            "support_burden":      _score_support_burden,
            "nps_detractor":       _score_nps,
            "csm_sentiment":       _score_csm_sentiment,
            "sdk_deprecated":      _score_sdk_deprecated,
            "engagement_decay":    _score_engagement_decay,
            "conflicting_signals": _score_conflicting_signals,
            "contract_urgency":    _score_contract_urgency,
        }

        total = 0.0
        breakdown = {}
        for key, scorer in scorers.items():
            sub_score, factor = scorer(row)
            weight = config.WEIGHTS[key]
            weighted = sub_score * weight
            total += weighted
            breakdown[key] = round(sub_score, 2)
            if factor:
                factor_list.append(factor)

        risk_score = min(round(total, 1), 100)

        if risk_score >= config.HIGH_RISK_THRESHOLD:
            tier = "High"
        elif risk_score >= config.MEDIUM_RISK_THRESHOLD:
            tier = "Medium"
        else:
            tier = "Low"

        results.append({
            "account_id":    row["account_id"],
            "risk_score":    risk_score,
            "risk_tier":     tier,
            "risk_factors":  factor_list,
            "score_breakdown": breakdown,
        })

    risk_df = pd.DataFrame(results)
    merged = features.merge(risk_df, on="account_id", how="left")
    return merged.sort_values("risk_score", ascending=False).reset_index(drop=True)
