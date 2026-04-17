"""
Pipeline — end-to-end orchestration of the renewal risk intelligence engine.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict

import pandas as pd

from .ingest import RawData, load_all
from .reconcile import reconcile_csm_notes, build_notes_by_account
from .features import build_features
from .risk import score_accounts
from .llm_explain import explain_all_accounts
from .insights import Insight, generate_all_insights, generate_portfolio_summary


@dataclass
class PipelineResult:
    """Complete output of the renewal risk pipeline."""
    scored_accounts: pd.DataFrame        # full feature + risk table
    insights: List[Insight]              # non-obvious insights
    portfolio_summary: str               # executive summary text
    raw_data: RawData                    # for drill-down views
    notes_by_account: Dict[int, str]     # reconciled CSM notes


def run_pipeline(use_llm: bool = True) -> PipelineResult:
    """
    Execute the full pipeline:
      1. Ingest all data sources
      2. Reconcile CSM notes → account IDs
      3. Build feature table (usage, support, NPS, notes, SDK)
      4. Score & tier all renewal-window accounts
      5. Generate LLM explanations
      6. Detect non-obvious insights
      7. Generate portfolio summary
    """
    # 1. Ingest
    raw = load_all()

    # 2. Reconcile CSM notes
    reconciled_notes = reconcile_csm_notes(raw.csm_notes, raw.accounts)
    notes_map = build_notes_by_account(reconciled_notes)

    # 3. Build features
    features = build_features(
        accounts=raw.accounts,
        usage=raw.usage,
        support=raw.support,
        nps=raw.nps,
        notes_by_account=notes_map,
    )

    # 4. Score accounts
    scored = score_accounts(features)

    # 5. LLM explanations
    scored = explain_all_accounts(scored, use_llm=use_llm)

    # 6. Insights
    insights = generate_all_insights(scored)

    # 7. Portfolio summary
    portfolio = generate_portfolio_summary(scored, insights, use_llm=use_llm)

    return PipelineResult(
        scored_accounts=scored,
        insights=insights,
        portfolio_summary=portfolio,
        raw_data=raw,
        notes_by_account=notes_map,
    )
