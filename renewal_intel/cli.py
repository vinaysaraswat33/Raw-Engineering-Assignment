"""
CLI interface — run the pipeline from the command line.
Usage:
    python -m renewal_intel.cli           # with Groq LLM
    python -m renewal_intel.cli --no-llm  # deterministic only
    python -m renewal_intel.cli --json-out results.json
"""
from __future__ import annotations
import argparse
import json
import sys

import pandas as pd

from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Renewal Risk Intelligence Engine",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Run without LLM (deterministic templates only)",
    )
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Path to write JSON output for downstream tools",
    )
    args = parser.parse_args()

    use_llm = not args.no_llm
    print("=" * 70)
    print("  RENEWAL RISK INTELLIGENCE ENGINE")
    print(f"  LLM: {'Groq (llama-3.3-70b-versatile)' if use_llm else 'Disabled (template mode)'}")
    print("=" * 70)
    print("\nRunning pipeline...\n")

    result = run_pipeline(use_llm=use_llm)

    # ── Print risk table ──────────────────────────────────────────────
    df = result.scored_accounts
    cols = ["account_id", "account_name", "arr", "risk_score", "risk_tier",
            "days_to_renewal", "plan_tier", "nps_score"]
    available = [c for c in cols if c in df.columns]

    print("\n" + "=" * 70)
    print("  RENEWAL RISK SCORECARD")
    print("=" * 70)

    for tier in ["High", "Medium", "Low"]:
        tier_df = df[df["risk_tier"] == tier]
        if tier_df.empty:
            continue
        print(f"\n{'🔴' if tier == 'High' else '🟡' if tier == 'Medium' else '🟢'} {tier.upper()} RISK ({len(tier_df)} accounts, ${tier_df['arr'].sum():,.0f} ARR)")
        print("-" * 70)
        for _, row in tier_df.iterrows():
            explanation = row.get("explanation", {})
            summary = explanation.get("summary", "No summary available") if isinstance(explanation, dict) else "No summary available"
            print(f"  [{row['risk_score']:5.1f}] {row['account_name']:<30s} ${row['arr']:>12,.0f}  {row['days_to_renewal']:3.0f}d")
            print(f"         {summary[:100]}")
            print()

    # ── Print insights ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  NON-OBVIOUS INSIGHTS")
    print("=" * 70)

    for insight in result.insights:
        print(f"\n{insight.title}")
        print(f"  {insight.description[:300]}")
        print(f"  Affected: {len(insight.affected_accounts)} accounts")
        print()

    # ── Print portfolio summary ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EXECUTIVE PORTFOLIO SUMMARY")
    print("=" * 70)
    print(f"\n{result.portfolio_summary}")

    # ── Optional JSON export ──────────────────────────────────────────
    if args.json_out:
        export = {
            "accounts": [],
            "insights": [],
            "portfolio_summary": result.portfolio_summary,
        }
        for _, row in df.iterrows():
            export["accounts"].append({
                "account_id": int(row["account_id"]),
                "account_name": row["account_name"],
                "arr": int(row["arr"]),
                "risk_score": float(row["risk_score"]),
                "risk_tier": row["risk_tier"],
                "days_to_renewal": int(row["days_to_renewal"]),
                "explanation": row.get("explanation", {}),
            })
        for ins in result.insights:
            export["insights"].append({
                "title": ins.title,
                "description": ins.description,
                "severity": ins.severity,
                "category": ins.category,
                "affected_accounts": ins.affected_accounts,
            })

        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, default=str)
        print(f"\n✅ JSON exported to {args.json_out}")


if __name__ == "__main__":
    main()
