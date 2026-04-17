"""
Non-obvious insight detection — surfaces patterns a simple rule-based system would miss.
"""
from __future__ import annotations
import json
import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import pandas as pd

from . import config


@dataclass
class Insight:
    """A single non-obvious insight."""
    title: str
    description: str
    affected_accounts: List[Dict]  # [{account_id, account_name, detail}, ...]
    severity: str  # "critical", "warning", "info"
    category: str  # "silent_churn", "nps_mismatch", "sdk_risk", "changelog_impact"


def detect_silent_churn(df: pd.DataFrame) -> Optional[Insight]:
    """
    Silent Churn: Accounts with decent NPS (≥7) but significant usage decline (>20%).
    These accounts 'like the team' but are actually leaving.
    The Meridian Health pattern from the CSM notes.
    """
    mask = (
        (df["nps_score"] >= 7) &
        ((df["api_trend"] < -20) | (df["user_trend"] < -20))
    )
    affected = df[mask]

    if affected.empty:
        return None

    accounts = []
    for _, row in affected.iterrows():
        accounts.append({
            "account_id": int(row["account_id"]),
            "account_name": row["account_name"],
            "nps_score": int(row["nps_score"]),
            "api_trend": f"{row['api_trend']:+.1f}%",
            "user_trend": f"{row['user_trend']:+.1f}%",
            "arr": f"${row['arr']:,.0f}",
        })

    total_arr = affected["arr"].sum()
    return Insight(
        title="🔇 Silent Churn Detected",
        description=(
            f"{len(accounts)} accounts show the 'silent churn' pattern: their NPS scores "
            f"are positive (≥7), suggesting satisfaction, but their actual product usage has "
            f"declined significantly (>20%). This means the contacts like your support team "
            f"but are quietly migrating away from the product. Total at-risk ARR: "
            f"${total_arr:,.0f}. A simple NPS-based system would miss these entirely."
        ),
        affected_accounts=accounts,
        severity="critical",
        category="silent_churn",
    )


def detect_nps_sentiment_mismatch(df: pd.DataFrame) -> Optional[Insight]:
    """
    NPS-Sentiment Mismatch: Score contradicts verbatim comment.
    E.g., NPS=2 but says "Best headless CMS" (data quality issue).
    Or NPS=10 but comment is negative.
    """
    positive_words = ["best", "love", "great", "phenomenal", "transformed", "recommend"]
    negative_words = ["downgrade", "wasted", "done", "forever", "fallen off", "disappointed", "steep"]

    mismatches = []
    for _, row in df.iterrows():
        nps = row.get("nps_score", -1)
        comment = str(row.get("nps_comment", "")).lower().strip()
        if not comment:
            continue

        has_positive = any(w in comment for w in positive_words)
        has_negative = any(w in comment for w in negative_words)

        # Low score but positive comment
        if nps <= 4 and has_positive and not has_negative:
            mismatches.append({
                "account_id": int(row["account_id"]),
                "account_name": row["account_name"],
                "nps_score": int(nps),
                "comment": row["nps_comment"],
                "mismatch_type": "Low score + positive comment (possible data entry error)",
            })
        # High score but negative comment
        elif nps >= 8 and has_negative and not has_positive:
            mismatches.append({
                "account_id": int(row["account_id"]),
                "account_name": row["account_name"],
                "nps_score": int(nps),
                "comment": row["nps_comment"],
                "mismatch_type": "High score + negative comment (possible survey fatigue)",
            })

    if not mismatches:
        return None

    return Insight(
        title="⚠️ NPS Score-Sentiment Mismatch",
        description=(
            f"{len(mismatches)} accounts have NPS scores that contradict their verbatim "
            f"comments. This suggests data quality issues — either survey fatigue (people "
            f"clicking random scores) or data entry errors. A naive model using NPS scores "
            f"alone would be misled by these."
        ),
        affected_accounts=mismatches,
        severity="warning",
        category="nps_mismatch",
    )


def detect_sdk_compound_risk(df: pd.DataFrame) -> Optional[Insight]:
    """
    SDK v3 Compound Risk: On deprecated SDK + high support tickets + approaching renewal.
    The changelog says v3 sunset is April 30, 2026. These accounts face forced migration
    AND are already struggling with support issues.
    """
    mask = (
        (df["sdk_is_v3"] == True) &
        (df["ticket_count"] > 3)
    )
    affected = df[mask]

    if affected.empty:
        return None

    accounts = []
    for _, row in affected.iterrows():
        accounts.append({
            "account_id": int(row["account_id"]),
            "account_name": row["account_name"],
            "sdk_version": row["latest_sdk"],
            "ticket_count": int(row["ticket_count"]),
            "p1_count": int(row["p1_count"]),
            "days_to_renewal": int(row["days_to_renewal"]),
            "arr": f"${row['arr']:,.0f}",
        })

    total_arr = affected["arr"].sum()
    return Insight(
        title="🔴 SDK v3 Deprecation Compound Risk",
        description=(
            f"{len(accounts)} accounts are still on SDK v3.x, which loses security patches "
            f"on April 30, 2026 (13 days away). These same accounts also have elevated "
            f"support ticket volumes, suggesting they're already struggling with the platform. "
            f"Forcing a major SDK migration on unhappy customers is a recipe for churn. "
            f"Total ARR at compound risk: ${total_arr:,.0f}. This insight emerges only by "
            f"cross-referencing the product changelog with per-account usage data and support history."
        ),
        affected_accounts=accounts,
        severity="critical",
        category="sdk_risk",
    )


def detect_changelog_impact(df: pd.DataFrame) -> Optional[Insight]:
    """
    Changelog Impact: Accounts affected by upcoming breaking changes.
    - Legacy editor removal (v4.4.0, May 2026)
    - REST API v2 sunset (April 30, 2026)
    - Accounts on SDK < v4.2.0 facing breaking response format change
    """
    affected = []
    for _, row in df.iterrows():
        sdk = str(row.get("latest_sdk", ""))
        issues = []

        # v3.x — REST API sunset + no security patches
        if sdk.startswith("v3."):
            issues.append("REST API v2 sunset (Apr 30) + end of security patches")

        # v4.0.0 or v4.1.0 — locale fallback bug, missing response format fix
        if sdk in ("v4.0.0", "v4.1.0"):
            issues.append("Missing locale fallback fix (v4.2.3) + response format change")

        if issues:
            affected.append({
                "account_id": int(row["account_id"]),
                "account_name": row["account_name"],
                "sdk_version": sdk,
                "impacts": issues,
                "arr": f"${row['arr']:,.0f}",
            })

    if not affected:
        return None

    return Insight(
        title="📋 Changelog-Driven Platform Risk",
        description=(
            f"{len(affected)} accounts are running SDK versions affected by upcoming "
            f"breaking changes documented in the product changelog. These include the REST "
            f"API v2 sunset, locale fallback bugs, and response format changes. Without "
            f"proactive outreach, these accounts will experience production breakages "
            f"around their renewal dates."
        ),
        affected_accounts=affected,
        severity="warning",
        category="changelog_impact",
    )


def detect_champion_risk(df: pd.DataFrame) -> Optional[Insight]:
    """
    Champion Risk: CSM notes mention key advocates at risk
    (nervous about their role, lost faith in roadmap, no-shows).
    """
    risk_phrases = [
        "lost faith", "nervous", "no show", "no-show",
        "reluctance", "on the fence", "biggest champion",
        "main advocate",
    ]

    affected = []
    for _, row in df.iterrows():
        note = str(row.get("csm_note_text", "")).lower()
        if not note:
            continue
        found = [p for p in risk_phrases if p in note]
        if found:
            affected.append({
                "account_id": int(row["account_id"]),
                "account_name": row["account_name"],
                "signals": found,
                "arr": f"${row['arr']:,.0f}",
            })

    if not affected:
        return None

    return Insight(
        title="👤 Champion Risk — Key Advocates at Risk",
        description=(
            f"{len(affected)} accounts have CSM notes indicating that key internal "
            f"champions are either disengaging, uncertain about their own roles, or "
            f"losing confidence in the product roadmap. Since renewals often depend on "
            f"one or two internal advocates, this is a high-signal risk indicator that "
            f"purely quantitative models would miss."
        ),
        affected_accounts=affected,
        severity="critical",
        category="champion_risk",
    )


def generate_all_insights(df: pd.DataFrame) -> List[Insight]:
    """Run all insight detectors and return results."""
    detectors = [
        detect_silent_churn,
        detect_nps_sentiment_mismatch,
        detect_sdk_compound_risk,
        detect_changelog_impact,
        detect_champion_risk,
    ]

    insights = []
    for detector in detectors:
        result = detector(df)
        if result is not None:
            insights.append(result)

    return insights


def generate_portfolio_summary(
    df: pd.DataFrame, insights: List[Insight], use_llm: bool = True,
) -> str:
    """
    Generate an executive portfolio summary using Groq LLM.
    Falls back to template if no API key.
    """
    high_risk = df[df["risk_tier"] == "High"]
    medium_risk = df[df["risk_tier"] == "Medium"]
    low_risk = df[df["risk_tier"] == "Low"]

    stats = {
        "total_accounts": len(df),
        "high_risk_count": len(high_risk),
        "medium_risk_count": len(medium_risk),
        "low_risk_count": len(low_risk),
        "total_arr": f"${df['arr'].sum():,.0f}",
        "high_risk_arr": f"${high_risk['arr'].sum():,.0f}",
        "medium_risk_arr": f"${medium_risk['arr'].sum():,.0f}",
        "avg_nps": round(df[df['nps_score'] >= 0]['nps_score'].mean(), 1) if (df['nps_score'] >= 0).any() else "N/A",
        "insight_count": len(insights),
    }

    insight_summaries = "\n".join(
        f"- {ins.title}: {ins.description[:200]}..."
        for ins in insights
    )

    if use_llm:
        try:
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                client = Groq(api_key=api_key)
                prompt = f"""You are a BizOps analyst writing an executive summary for the quarterly renewal review.

PORTFOLIO STATS:
{json.dumps(stats, indent=2)}

KEY INSIGHTS:
{insight_summaries}

HIGH-RISK ACCOUNTS (top 5 by ARR):
{chr(10).join(f"- {r['account_name']} (${r['arr']:,.0f} ARR, score {r['risk_score']}/100)" for _, r in high_risk.head(5).iterrows())}

Write a 3-4 paragraph executive summary covering:
1. Overall portfolio health and key numbers
2. Most critical risks requiring immediate attention
3. Non-obvious patterns the team should watch
4. Recommended focus areas for the next 2 weeks

Keep it concise, data-driven, and actionable. Write in plain English for a VP audience."""

                response = client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a concise, data-driven BizOps analyst."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=800,
                    temperature=0.4,
                )
                return response.choices[0].message.content
        except Exception:
            pass

    # Template fallback
    return (
        f"**Quarterly Renewal Portfolio Summary**\n\n"
        f"Of {stats['total_accounts']} accounts renewing in the next 90 days "
        f"(total ARR: {stats['total_arr']}), **{stats['high_risk_count']} are High Risk** "
        f"representing {stats['high_risk_arr']} in ARR, and "
        f"**{stats['medium_risk_count']} are Medium Risk** ({stats['medium_risk_arr']} ARR).\n\n"
        f"The average NPS among renewing accounts is {stats['avg_nps']}. "
        f"Our analysis surfaced {stats['insight_count']} non-obvious insights including "
        f"silent churn patterns, NPS data quality issues, and compounding SDK deprecation risks.\n\n"
        f"**Immediate priorities:**\n"
        f"1. Address the {stats['high_risk_count']} High-Risk accounts with tailored action plans\n"
        f"2. Resolve open P1 tickets for at-risk accounts before renewal conversations\n"
        f"3. Proactively reach out to accounts on deprecated SDK v3.x (sunset April 30)\n"
        f"4. Investigate silent churn signals — accounts with good NPS but declining usage"
    )
