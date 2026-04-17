"""
LLM explanation layer — Groq-powered plain-English explanations for at-risk accounts.
Falls back to deterministic templates when GROQ_API_KEY is not set.
"""
from __future__ import annotations
import json
import os
import time
from typing import Dict, List, Optional

import pandas as pd

from . import config


def _get_groq_client():
    """Lazy-import and create Groq client."""
    try:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None
        return Groq(api_key=api_key)
    except ImportError:
        return None


def _build_account_prompt(row: pd.Series) -> str:
    """Build the prompt for a single account's explanation."""
    factors = row.get("risk_factors", [])
    factors_text = "\n".join(f"  - {f}" for f in factors) if factors else "  (none identified)"

    csm_text = row.get("csm_note_text", "")
    csm_snippet = csm_text[:500] if csm_text else "(No CSM notes available)"

    nps_comment = row.get("nps_comment", "")
    nps_info = f"NPS Score: {row.get('nps_score', 'N/A')}"
    if nps_comment:
        nps_info += f"\nNPS Verbatim: \"{nps_comment}\""

    return f"""You are a BizOps renewal risk analyst at Contentstack (a headless CMS company).
Analyze this account and provide a brief, actionable risk assessment.

ACCOUNT: {row.get('account_name', 'Unknown')} (ID: {row.get('account_id')})
Plan: {row.get('plan_tier', 'N/A')} | ARR: ${row.get('arr', 0):,.0f} | Industry: {row.get('industry', 'N/A')}
Region: {row.get('region', 'N/A')} | CSM: {row.get('csm_name', 'N/A')}
Contract End: {row.get('contract_end_date')} | Days to Renewal: {row.get('days_to_renewal', 'N/A')}

RISK SCORE: {row.get('risk_score', 0)}/100 — {row.get('risk_tier', 'N/A')} Risk

KEY RISK FACTORS:
{factors_text}

USAGE TRENDS (6-month):
  API Calls: {row.get('api_trend', 0):+.1f}% | Active Users: {row.get('user_trend', 0):+.1f}%
  Content Created: {row.get('content_trend', 0):+.1f}% | Workflows: {row.get('workflow_trend', 0):+.1f}%
  SDK Version: {row.get('latest_sdk', 'N/A')}

SUPPORT:
  Total Tickets: {row.get('ticket_count', 0)} | Open: {row.get('open_tickets', 0)} | P1: {row.get('p1_count', 0)}
  Escalated: {row.get('escalated_tickets', 0)} | Recurring: {row.get('recurring_count', 0)}

{nps_info}

CSM NOTES:
{csm_snippet}

Respond in this JSON format:
{{
  "summary": "2-3 sentence plain-English summary of why this account is at risk (or healthy)",
  "top_signals": ["signal 1", "signal 2", "signal 3"],
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "data_conflicts": ["any contradictions in the data, e.g. 'NPS is high but usage is declining'"],
  "urgency": "immediate|this_week|this_month|routine"
}}

Be specific and reference actual data points. Keep it concise and actionable."""


def _template_explanation(row: pd.Series) -> Dict:
    """Deterministic fallback when no LLM API key is available."""
    factors = row.get("risk_factors", [])
    tier = row.get("risk_tier", "Low")
    score = row.get("risk_score", 0)
    name = row.get("account_name", "Unknown")
    arr = row.get("arr", 0)

    # Build summary from factors
    if tier == "High":
        summary = f"{name} (${arr:,.0f} ARR) is a high-risk renewal with a score of {score}/100. "
        if factors:
            summary += f"Key concerns: {factors[0].lower()}"
            if len(factors) > 1:
                summary += f", and {factors[1].lower()}"
            summary += "."
    elif tier == "Medium":
        summary = f"{name} (${arr:,.0f} ARR) shows moderate renewal risk ({score}/100). "
        if factors:
            summary += f"Watch for: {factors[0].lower()}."
    else:
        summary = f"{name} (${arr:,.0f} ARR) appears healthy for renewal ({score}/100)."

    # Top signals
    top_signals = factors[:3] if factors else ["No significant risk signals detected"]

    # Actions
    actions = []
    if row.get("competitor_mentioned", False):
        actions.append("Schedule executive sponsor call to address competitive evaluation")
    if row.get("sdk_is_v3", False):
        actions.append("Prioritize SDK v3→v4 migration support before April 30 deadline")
    if row.get("open_tickets", 0) > 0:
        actions.append(f"Escalate {int(row['open_tickets'])} open support tickets for resolution")
    if row.get("nps_score", 10) <= 6 and row.get("nps_score", 10) >= 0:
        actions.append("Schedule NPS detractor recovery call with account team")
    if row.get("api_trend", 0) < -20:
        actions.append("Investigate usage decline — potential adoption or value gap")
    if not actions:
        actions.append("Continue standard renewal cadence — no blockers identified")

    # Data conflicts
    conflicts = []
    nps = row.get("nps_score", -1)
    api_trend = row.get("api_trend", 0)
    if nps >= 7 and api_trend < -15:
        conflicts.append(f"NPS is {nps} (positive) but API usage declined {abs(api_trend):.0f}% — possible silent churn")
    if nps <= 4 and api_trend > 10:
        conflicts.append(f"NPS is very low ({nps}) but usage is growing — sentiment vs. behavior mismatch")

    # Urgency
    days = row.get("days_to_renewal", 90)
    if tier == "High" and days <= 30:
        urgency = "immediate"
    elif tier == "High":
        urgency = "this_week"
    elif tier == "Medium":
        urgency = "this_month"
    else:
        urgency = "routine"

    return {
        "summary": summary,
        "top_signals": top_signals,
        "recommended_actions": actions[:3],
        "data_conflicts": conflicts,
        "urgency": urgency,
    }


def explain_account(row: pd.Series, client=None) -> Dict:
    """Get LLM explanation for a single account, with fallback."""
    if client is None:
        return _template_explanation(row)

    prompt = _build_account_prompt(row)
    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a BizOps renewal risk analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.GROQ_MAX_TOKENS,
            temperature=config.GROQ_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        # Fallback on any error
        result = _template_explanation(row)
        result["_llm_error"] = str(e)
        return result


def explain_all_accounts(df: pd.DataFrame, use_llm: bool = True) -> pd.DataFrame:
    """
    Generate explanations for all accounts.
    Adds 'explanation' column with dict of summary, signals, actions.
    """
    client = _get_groq_client() if use_llm else None

    explanations = []
    for _, row in df.iterrows():
        explanation = explain_account(row, client)
        explanations.append(explanation)
        # Rate limiting for Groq free tier
        if client is not None:
            time.sleep(0.5)

    df = df.copy()
    df["explanation"] = explanations
    return df
