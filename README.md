# Renewal Risk Intelligence Engine

> **Take-Home Assignment** — Applied AI Engineer, BizOps @ Contentstack  
> Prototype that ingests messy renewal signals, **scores accounts**, and produces **plain-English AI explanations** with recommended actions.

---

## 🎯 Approach

### Problem Definition
"At risk" = any account showing **two or more converging negative signals**. A single bad NPS or minor usage dip alone isn't alarming; but usage decline + competitor mention in CSM notes + open P1 tickets = clear risk.

### Pipeline Architecture

```
DATA SOURCES → INGEST → RECONCILE → FEATURES → RISK SCORE → LLM EXPLAIN → INSIGHTS
     │                                                                         │
     └─── data/accounts.csv, data/usage_metrics.csv, data/support_tickets.csv,               │
          data/nps_responses.csv, data/csm_notes.txt, data/changelog.md                       │
                                                                               ▼
                                                                     STREAMLIT DASHBOARD
```

### Six Steps:

1. **Ingest** — Load all 6 data files. Parse CSM notes into blocks (split on `---`). Parse changelog into structured entries with version, category, and deprecation dates.

2. **Reconcile** — CSM notes don't cleanly map to accounts. We use:
   - **Regex extraction** for explicit IDs (`acct 1001`, `#1007`, `(1004)`, `account 1016`)
   - **RapidFuzz** (`token_sort_ratio`) for fuzzy name matching to handle typos: "BritePath" → BrightPath Solutions, "Pinacle" → Pinnacle Media Group, "Thunderbolt Moters" → Thunderbolt Motors

3. **Feature Engineering** — Compute per-account signals from every data source:
   - **Usage**: API call trend, active user trend, content creation trend, workflow trend (early 3 months vs late 3 months)
   - **Support**: Total/open/P1/escalated tickets, recurring issues, recent ticket velocity
   - **NPS**: Score + verbatim sentiment analysis
   - **CSM Notes**: Competitor mentions, negative keywords, champion risk phrases
   - **SDK Version**: Flag v3.x users facing April 30 deprecation deadline
   - **Changelog Cross-Reference**: Breaking changes affecting specific SDK versions

4. **Risk Scoring** — Weighted composite score (0–100) from 8 sub-scores:

   | Component | Weight | What It Measures |
   |-----------|--------|------------------|
   | Usage Decline | 20 pts | API + content trend |
   | Support Burden | 15 pts | Open/P1/escalated tickets |
   | NPS Detractor | 15 pts | Survey score |
   | CSM Sentiment | 15 pts | Competitor mentions, negative keywords |
   | SDK Deprecated | 10 pts | v3.x facing sunset |
   | Engagement Decay | 10 pts | Active user decline |
   | Conflicting Signals | 10 pts | Silent churn (good NPS + bad usage) |
   | Contract Urgency | 5 pts | Days to renewal |

   Tiers: **High ≥ 65** | **Medium 40–64** | **Low < 40**

5. **LLM Explanations** — Groq API (`llama-3.3-70b-versatile`) generates:
   - Plain-English risk summary per account
   - Top 3 risk signals
   - Recommended actions for account teams
   - Data conflict callouts (e.g., "NPS is positive but usage is declining")
   - Falls back to intelligent deterministic templates when no API key is set

6. **Non-Obvious Insights** — Five pattern detectors that surface what rule-based systems miss:
   - **🔇 Silent Churn**: NPS ≥ 7 but usage down >20% (the Meridian Health pattern)
   - **⚠️ NPS-Sentiment Mismatch**: Score contradicts verbatim (e.g., NPS=2 + "Best CMS ever")
   - **🔴 SDK Compound Risk**: Deprecated SDK + high tickets + approaching renewal
   - **📋 Changelog Impact**: Cross-reference changelog deprecations with per-account SDK versions
   - **👤 Champion Risk**: CSM notes mentioning at-risk advocates ("lost faith", "nervous")

---

## 🏗️ Why the Changelog Matters

The changelog reveals critical platform risks that are invisible in isolation:
- **SDK v3.x sunset** (April 30, 2026) — accounts still on v3 will lose security patches
- **Breaking API response format** (v4.2.0) — apps using `response.entry` must update to `response.data`
- **Legacy editor removal** (v4.4.0, May 2026) — forced migration for all users
- **Locale fallback fix** (v4.2.3) — only available in newer SDK versions

Cross-referencing `sdk_version` from usage data with changelog entries surfaces **compound risk**: an account on deprecated v3.x, with high P1 ticket volume, approaching renewal while the team is scrambling to migrate before the deadline.

---

## ⚖️ Tradeoffs

| Decision | Rationale |
|----------|-----------|
| **Weighted rules vs. ML model** | Interpretable, auditable, controllable. BizOps teams need to *explain* risk to account managers. Would add calibration + backtesting with real historical churn labels in production. |
| **Fuzzy matching vs. embeddings** | RapidFuzz is fast, deterministic, and sufficient for the ~30 note blocks. At scale (1000+ notes), would use sentence embeddings + vector store. |
| **Groq free tier vs. GPT-4** | Groq provides fast, free inference with LLaMA 3.3 70B. Quality is comparable for structured JSON extraction tasks. Would benchmark against Claude/GPT-4 in production. |
| **Template fallback** | Always works without API keys — critical for demos and CI/CD. |
| **Fixed reference date** | `2026-04-17` in config.py matches the synthetic dataset. Parameterize for production. |

---

## 🚀 With More Time

- **Supervised model**: Train on historical churn labels with survival analysis
- **Embeddings**: Vector store for CSM notes at scale with per-field lineage
- **Real-time**: Salesforce/HubSpot write-back, Slack digest alerts, webhook triggers
- **Multi-language NPS**: Dedicated translation pipeline for Chinese/French/Spanish verbatims
- **Role-based views**: CS managers see action plans, Finance sees ARR impact, execs see portfolio summary
- **Backtesting**: Validate scoring weights against historical renewal outcomes

---

## 🖥️ How to Run

### Prerequisites
- Python 3.10+
- (Optional) [Groq API key](https://console.groq.com) for AI explanations

### Setup

```bash
cd renewal_intelligence_takehome

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# (Optional) Set Groq API key for LLM features
set GROQ_API_KEY=gsk_your_key_here        # Windows
# export GROQ_API_KEY=gsk_your_key_here   # macOS/Linux
```

### CLI

```bash
# With Groq LLM
python -m renewal_intel.cli

# Without LLM (deterministic templates)
python -m renewal_intel.cli --no-llm

# Export JSON
python -m renewal_intel.cli --no-llm --json-out results.json
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

---

## 📁 Project Layout

```
renewal_intelligence_takehome/
  app.py                     # Streamlit dashboard (5 pages)
  requirements.txt
  README.md
  ASSIGNMENT.md              # Original assignment brief

  # Data directory
  data/
    accounts.csv               # 120 accounts with contract details
    usage_metrics.csv          # 6 months of product usage
    support_tickets.csv        # Support ticket history
    nps_responses.csv          # NPS survey responses
    csm_notes.txt              # Unstructured CSM call notes (messy)
    changelog.md               # Product changelog (Q4 2025 – Q1 2026)

  # Core engine
  renewal_intel/
    __init__.py
    config.py                # Reference date, paths, weights, keywords
    ingest.py                # Load & parse all data sources
    reconcile.py             # Fuzzy match CSM notes → account IDs
    features.py              # Feature engineering from all signals
    risk.py                  # Weighted risk scoring (0-100)
    llm_explain.py           # Groq LLM explanations + template fallback
    insights.py              # Non-obvious insight detectors
    pipeline.py              # End-to-end orchestration
    cli.py                   # Command-line interface
```

---

## 🤖 LLM Usage — Meaningful, Not Gimmick

The LLM (Groq / LLaMA 3.3 70B) serves three distinct purposes:

1. **Per-Account Narratives**: Receives all computed signals (usage, support, NPS, notes, SDK) and produces contextual plain-English summaries with recommended actions. This is more than summarization — the LLM identifies *relationships* between signals (e.g., "usage decline coincides with the workflow automation bug").

2. **Data Conflict Detection**: The LLM explicitly calls out contradictions (e.g., "NPS is 8 but CSM notes mention competitor evaluation — trust the notes, not the score").

3. **Portfolio Synthesis**: Generates an executive summary across the entire at-risk cohort, identifying themes and priority areas — something that's hard to template deterministically.

The scoring itself is **deterministic and interpretable** — the LLM explains decisions, it doesn't make them.

---

*Built for the Contentstack BizOps Take-Home Assignment — April 2026*
