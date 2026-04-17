"""
Configuration — reference dates, file paths, model settings, scoring weights.
"""
from pathlib import Path
from datetime import date, timedelta

# ── Reference date (today for the synthetic dataset) ──────────────────────
REFERENCE_DATE = date(2026, 4, 17)
RENEWAL_WINDOW_DAYS = 90  # look-ahead window for upcoming renewals
RENEWAL_CUTOFF = REFERENCE_DATE + timedelta(days=RENEWAL_WINDOW_DAYS)

# ── Data directory ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ACCOUNTS_CSV      = DATA_DIR / "accounts.csv"
USAGE_CSV          = DATA_DIR / "usage_metrics.csv"
SUPPORT_CSV        = DATA_DIR / "support_tickets.csv"
NPS_CSV            = DATA_DIR / "nps_responses.csv"
CSM_NOTES_TXT      = DATA_DIR / "csm_notes.txt"
CHANGELOG_MD       = DATA_DIR / "changelog.md"

# ── Groq LLM settings ────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS = 1024
GROQ_TEMPERATURE = 0.3  # low for deterministic, factual output

# ── Risk scoring weights (sum = 100) ─────────────────────────────────────
WEIGHTS = {
    "usage_decline":      20,
    "support_burden":     15,
    "nps_detractor":      15,
    "csm_sentiment":      15,
    "sdk_deprecated":     10,
    "engagement_decay":   10,
    "conflicting_signals": 10,
    "contract_urgency":    5,
}

# ── Tier thresholds ───────────────────────────────────────────────────────
HIGH_RISK_THRESHOLD   = 65
MEDIUM_RISK_THRESHOLD = 40

# ── Changelog-derived dates ───────────────────────────────────────────────
SDK_V3_SUNSET_DATE       = date(2026, 4, 30)
LEGACY_EDITOR_REMOVAL    = "v4.4.0 — May 2026"
REST_API_V2_SUNSET       = date(2026, 4, 30)

# ── Competitor keywords (for CSM note scanning) ──────────────────────────
COMPETITOR_KEYWORDS = [
    "hygraph", "contentful", "strapi", "sanity", "kontent.ai",
    "builder.io", "wordpress", "drupal", "competitor", "evaluate",
    "exploring options", "explore options",
]

# ── Negative sentiment keywords ──────────────────────────────────────────
NEGATIVE_KEYWORDS = [
    "frustrated", "furious", "threatened", "walk", "done",
    "lost faith", "dealbreaker", "embarrassing", "reluctance",
    "tense", "no show", "missed qbr", "budget cut", "shelfware",
    "downgrade", "churn", "silent churn", "nervous",
]
