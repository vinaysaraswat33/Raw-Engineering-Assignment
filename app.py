"""
Renewal Risk Intelligence Engine — Streamlit Dashboard
Premium UI with executive dashboard, account deep-dive, and insights panel.
"""
import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Renewal Risk Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header */
    .main-header {
        background: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .main-header h1 {
        color: #1d3557;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #64748b;
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
    }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1d3557;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* Risk tier badges */
    .tier-high {
        background: #fee2e2;
        color: #b91c1c;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .tier-medium {
        background: #fef3c7;
        color: #b45309;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .tier-low {
        background: #dcfce7;
        color: #15803d;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* Insight cards */
    .insight-card {
        background: #ffffff;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
    .insight-card.warning { border-left-color: #f59e0b; }
    .insight-card.info { border-left-color: #3b82f6; }
    .insight-card h4 {
        margin: 0 0 0.5rem 0;
        color: #1e293b;
    }
    .insight-card p {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Account detail */
    .detail-section {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    .detail-section h4 {
        color: #1d3557;
        margin-top: 0;
        font-size: 1rem;
    }

    /* Factor pills */
    .factor-pill {
        background: #f1f5f9;
        color: #475569;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
        border: 1px solid #cbd5e1;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} - Removed to allow sidebar toggle button to appear */
    .dataframe { font-size: 0.85rem; }
    
    /* Make SURE sidebar background gradient is gone! */
    section[data-testid="stSidebar"] {
        background: transparent !important;
    }

</style>
""", unsafe_allow_html=True)


# ── Cache pipeline execution ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_cached_pipeline(use_llm: bool):
    """Run pipeline and cache results."""
    from renewal_intel.pipeline import run_pipeline
    result = run_pipeline(use_llm=use_llm)
    # Serialize insights for caching
    insights_data = []
    for ins in result.insights:
        insights_data.append({
            "title": ins.title,
            "description": ins.description,
            "affected_accounts": ins.affected_accounts,
            "severity": ins.severity,
            "category": ins.category,
        })
    return result.scored_accounts, insights_data, result.portfolio_summary, result.notes_by_account


def get_tier_badge(tier: str) -> str:
    """Return HTML badge for risk tier."""
    cls = f"tier-{tier.lower()}"
    return f'<span class="{cls}">{tier}</span>'


def format_arr(val) -> str:
    """Format ARR value."""
    if pd.isna(val):
        return "N/A"
    return f"${val:,.0f}"


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    has_key = bool(groq_key)

    use_llm = st.toggle(
        "Enable Groq LLM",
        value=has_key,
        help="Uses Groq (llama-3.3-70b-versatile) for AI-powered explanations. Requires GROQ_API_KEY env var.",
        disabled=not has_key,
    )

    if not has_key:
        st.info("💡 Set `GROQ_API_KEY` env var to enable AI explanations. Running in template mode.")

    st.divider()

    st.markdown("## 📊 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Executive Dashboard", "📋 Risk Scorecard", "🔍 Account Deep Dive",
         "💡 Non-Obvious Insights", "🏗️ Architecture"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        '<p style="color:#666;font-size:0.75rem;text-align:center;">'
        'Renewal Risk Intelligence Engine v1.0<br>'
        'Powered by Groq + LLaMA 3.3<br>'
        '© 2026 Contentstack BizOps'
        '</p>',
        unsafe_allow_html=True,
    )


# ── Load data ─────────────────────────────────────────────────────────────
with st.spinner("🔄 Running Renewal Intelligence Pipeline..."):
    df, insights_data, portfolio_summary, notes_map = run_cached_pipeline(use_llm)

high_risk = df[df["risk_tier"] == "High"]
medium_risk = df[df["risk_tier"] == "Medium"]
low_risk = df[df["risk_tier"] == "Low"]


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: Executive Dashboard
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Dashboard":
    # Header
    st.markdown('''
    <div class="main-header">
        <h1>🔍 Renewal Risk Intelligence</h1>
        <p>Q2 2026 Renewal Cohort — AI-Powered Risk Assessment</p>
    </div>
    ''', unsafe_allow_html=True)

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-value">{len(df)}</div>
            <div class="kpi-label">Accounts Renewing</div>
        </div>
        ''', unsafe_allow_html=True)

    with c2:
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-value">{len(high_risk)}</div>
            <div class="kpi-label">High Risk</div>
        </div>
        ''', unsafe_allow_html=True)

    with c3:
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-value">${high_risk["arr"].sum():,.0f}</div>
            <div class="kpi-label">At-Risk ARR</div>
        </div>
        ''', unsafe_allow_html=True)

    with c4:
        avg_nps = df[df["nps_score"] >= 0]["nps_score"].mean()
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-value">{avg_nps:.1f}</div>
            <div class="kpi-label">Avg NPS (Cohort)</div>
        </div>
        ''', unsafe_allow_html=True)

    with c5:
        st.markdown(f'''
        <div class="kpi-card">
            <div class="kpi-value">{len(insights_data)}</div>
            <div class="kpi-label">Key Insights</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        # Risk distribution donut
        tier_counts = df["risk_tier"].value_counts()
        fig_donut = go.Figure(data=[go.Pie(
            labels=tier_counts.index,
            values=tier_counts.values,
            hole=0.55,
            marker_colors=["#dc2626", "#f59e0b", "#16a34a"],
            textinfo="label+value",
            textfont=dict(size=14, color="white"),
            sort=False,
        )])
        fig_donut.update_layout(
            title=dict(text="Risk Distribution", font=dict(size=16, color="#1e293b")),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            font=dict(color="#475569"),
            legend=dict(font=dict(color="#475569")),
            height=350,
            margin=dict(t=60, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        # ARR by tier bar chart
        tier_arr = df.groupby("risk_tier")["arr"].sum().reindex(["High", "Medium", "Low"])
        fig_bar = go.Figure(data=[go.Bar(
            x=tier_arr.index,
            y=tier_arr.values,
            marker_color=["#dc2626", "#f59e0b", "#16a34a"],
            text=[f"${v:,.0f}" for v in tier_arr.values],
            textposition="auto",
            textfont=dict(color="white", size=13),
        )])
        fig_bar.update_layout(
            title=dict(text="ARR at Risk by Tier", font=dict(size=16, color="#1e293b")),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            font=dict(color="#475569"),
            yaxis=dict(gridcolor="#e2e8f0", title="ARR ($)"),
            xaxis=dict(title=""),
            height=350,
            margin=dict(t=60, b=40, l=60, r=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Second charts row
    col3, col4 = st.columns(2)

    with col3:
        # Top 10 at-risk accounts
        top10 = df.head(10)[["account_name", "risk_score", "arr", "risk_tier"]].copy()
        colors = top10["risk_tier"].map({"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"})
        fig_top = go.Figure(data=[go.Bar(
            y=top10["account_name"],
            x=top10["risk_score"],
            orientation="h",
            marker_color=colors,
            text=[f'{s:.0f}' for s in top10["risk_score"]],
            textposition="auto",
            textfont=dict(color="white", size=12),
        )])
        fig_top.update_layout(
            title=dict(text="Top 10 At-Risk Accounts", font=dict(size=16, color="#1e293b")),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            font=dict(color="#475569"),
            xaxis=dict(gridcolor="#e2e8f0", title="Risk Score", range=[0, 100]),
            yaxis=dict(autorange="reversed"),
            height=400,
            margin=dict(t=60, b=40, l=180, r=20),
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col4:
        # Risk score by industry
        industry_risk = df.groupby("industry")["risk_score"].mean().sort_values(ascending=False).head(10)
        fig_ind = go.Figure(data=[go.Bar(
            y=industry_risk.index,
            x=industry_risk.values,
            orientation="h",
            marker_color="#dc2626",
            text=[f'{s:.0f}' for s in industry_risk.values],
            textposition="auto",
            textfont=dict(color="white", size=12),
        )])
        fig_ind.update_layout(
            title=dict(text="Avg Risk Score by Industry", font=dict(size=16, color="#1e293b")),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
            font=dict(color="#475569"),
            xaxis=dict(gridcolor="#e2e8f0", title="Avg Risk Score", range=[0, 100]),
            yaxis=dict(autorange="reversed"),
            height=400,
            margin=dict(t=60, b=40, l=180, r=20),
        )
        st.plotly_chart(fig_ind, use_container_width=True)

    # Portfolio Summary
    st.markdown("---")
    st.markdown("### 📝 Executive Portfolio Summary")
    st.markdown(portfolio_summary)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: Risk Scorecard
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📋 Risk Scorecard":
    st.markdown('''
    <div class="main-header">
        <h1>📋 Risk Scorecard</h1>
        <p>All accounts renewing in the next 90 days — sorted by risk score</p>
    </div>
    ''', unsafe_allow_html=True)

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        tier_filter = st.multiselect("Risk Tier", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    with fc2:
        industry_filter = st.multiselect("Industry", sorted(df["industry"].unique()), default=[])
    with fc3:
        region_filter = st.multiselect("Region", sorted(df["region"].unique()), default=[])
    with fc4:
        plan_filter = st.multiselect("Plan", sorted(df["plan_tier"].unique()), default=[])

    filtered = df[df["risk_tier"].isin(tier_filter)]
    if industry_filter:
        filtered = filtered[filtered["industry"].isin(industry_filter)]
    if region_filter:
        filtered = filtered[filtered["region"].isin(region_filter)]
    if plan_filter:
        filtered = filtered[filtered["plan_tier"].isin(plan_filter)]

    st.markdown(f"**Showing {len(filtered)} of {len(df)} accounts** | Total ARR: **{format_arr(filtered['arr'].sum())}**")

    # Display table
    display_cols = [
        "risk_tier", "risk_score", "account_name", "arr", "plan_tier",
        "industry", "region", "days_to_renewal", "nps_score",
        "api_trend", "user_trend", "ticket_count", "open_tickets",
        "p1_count", "latest_sdk",
    ]
    available = [c for c in display_cols if c in filtered.columns]

    styled_df = filtered[available].copy()
    styled_df["arr"] = styled_df["arr"].apply(lambda x: f"${x:,.0f}")
    styled_df["api_trend"] = styled_df["api_trend"].apply(lambda x: f"{x:+.1f}%")
    styled_df["user_trend"] = styled_df["user_trend"].apply(lambda x: f"{x:+.1f}%")
    styled_df.columns = [
        "Tier", "Score", "Account", "ARR", "Plan",
        "Industry", "Region", "Days", "NPS",
        "API Δ", "Users Δ", "Tickets", "Open",
        "P1s", "SDK",
    ]

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=600,
        hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: Account Deep Dive
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Account Deep Dive":
    st.markdown('''
    <div class="main-header">
        <h1>🔍 Account Deep Dive</h1>
        <p>Detailed view of individual account risk signals</p>
    </div>
    ''', unsafe_allow_html=True)

    # Account selector
    account_options = [
        f"[{row['risk_tier']}] {row['account_name']} — {row['risk_score']:.0f}/100"
        for _, row in df.iterrows()
    ]
    selected_idx = st.selectbox("Select Account", range(len(account_options)),
                                format_func=lambda i: account_options[i])
    row = df.iloc[selected_idx]

    tier = row["risk_tier"]
    badge_color = {"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"}[tier]
    badge_fill = {"High": "rgba(220,38,38,0.2)", "Medium": "rgba(245,158,11,0.2)", "Low": "rgba(22,163,74,0.2)"}[tier]

    # Header card
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.metric("Risk Score", f"{row['risk_score']:.0f}/100")
    with h2:
        st.metric("ARR", format_arr(row["arr"]))
    with h3:
        st.metric("Days to Renewal", f"{int(row['days_to_renewal'])}")
    with h4:
        st.metric("NPS Score", f"{int(row['nps_score'])}" if row['nps_score'] >= 0 else "N/A")

    st.markdown(f"**{row['account_name']}** | {row['plan_tier']} | {row['industry']} | {row['region']} | CSM: {row['csm_name']}")
    st.markdown(f"Risk Tier: **:{badge_color}[{tier}]** | Contract End: **{row['contract_end_date']}** | SDK: **{row.get('latest_sdk', 'N/A')}**")

    st.markdown("---")

    # Two columns: explanation + factors
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("#### 🤖 AI Risk Assessment")
        explanation = row.get("explanation", {})
        if isinstance(explanation, dict):
            st.markdown(f"**Summary:** {explanation.get('summary', 'N/A')}")

            signals = explanation.get("top_signals", [])
            if signals:
                st.markdown("**Top Signals:**")
                for s in signals:
                    st.markdown(f"- 🔸 {s}")

            actions = explanation.get("recommended_actions", [])
            if actions:
                st.markdown("**Recommended Actions:**")
                for a in actions:
                    st.markdown(f"- ✅ {a}")

            conflicts = explanation.get("data_conflicts", [])
            if conflicts:
                st.markdown("**⚠️ Data Conflicts:**")
                for c in conflicts:
                    st.markdown(f"- {c}")

            urgency = explanation.get("urgency", "routine")
            urgency_emoji = {"immediate": "🚨", "this_week": "⚡", "this_month": "📅", "routine": "✅"}
            st.markdown(f"**Urgency:** {urgency_emoji.get(urgency, '📋')} {urgency.replace('_', ' ').title()}")

    with c2:
        st.markdown("#### 📊 Score Breakdown")
        breakdown = row.get("score_breakdown", {})
        if isinstance(breakdown, dict) and breakdown:
            labels = list(breakdown.keys())
            values = list(breakdown.values())
            from renewal_intel.config import WEIGHTS
            weighted = [v * WEIGHTS.get(k, 0) for k, v in breakdown.items()]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=[k.replace("_", " ").title() for k in labels],
                fill="toself",
                line_color=badge_color,
                fillcolor=badge_fill,
                name="Risk Sub-Scores",
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(255,255,255,1)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#cbd5e1"),
                    angularaxis=dict(gridcolor="#cbd5e1"),
                ),
                paper_bgcolor="rgba(255,255,255,1)",
                font=dict(color="#475569", size=10),
                height=300,
                margin=dict(t=20, b=20, l=80, r=80),
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # Usage trends
    st.markdown("#### 📈 Usage Trends (6-Month)")
    from renewal_intel.ingest import load_usage
    usage_data = load_usage()
    acct_usage = usage_data[usage_data["account_id"] == row["account_id"]].sort_values("month")

    if not acct_usage.empty:
        uc1, uc2 = st.columns(2)
        with uc1:
            fig_usage = make_subplots(specs=[[{"secondary_y": True}]])
            fig_usage.add_trace(
                go.Scatter(x=acct_usage["month"], y=acct_usage["api_calls"],
                          name="API Calls", line=dict(color="#dc2626", width=2)),
                secondary_y=False,
            )
            fig_usage.add_trace(
                go.Scatter(x=acct_usage["month"], y=acct_usage["active_users"],
                          name="Active Users", line=dict(color="#2563eb", width=2, dash="dot")),
                secondary_y=True,
            )
            fig_usage.update_layout(
                title=dict(text="API Calls & Active Users", font=dict(size=14, color="#1e293b")),
                paper_bgcolor="rgba(255,255,255,1)", plot_bgcolor="rgba(255,255,255,1)",
                font=dict(color="#475569"),
                yaxis=dict(gridcolor="#e2e8f0"),
                yaxis2=dict(gridcolor="#e2e8f0"),
                height=280, margin=dict(t=50, b=40, l=50, r=50),
                legend=dict(y=1.15, orientation="h"),
            )
            st.plotly_chart(fig_usage, use_container_width=True)

        with uc2:
            fig_content = go.Figure()
            fig_content.add_trace(go.Bar(
                x=acct_usage["month"], y=acct_usage["content_entries_created"],
                name="Content Created", marker_color="#16a34a",
            ))
            fig_content.add_trace(go.Scatter(
                x=acct_usage["month"], y=acct_usage["workflows_triggered"],
                name="Workflows", line=dict(color="#f59e0b", width=2),
            ))
            fig_content.update_layout(
                title=dict(text="Content Created & Workflows", font=dict(size=14, color="#1e293b")),
                paper_bgcolor="rgba(255,255,255,1)", plot_bgcolor="rgba(255,255,255,1)",
                font=dict(color="#475569"),
                yaxis=dict(gridcolor="#e2e8f0"),
                height=280, margin=dict(t=50, b=40, l=50, r=50),
                legend=dict(y=1.15, orientation="h"),
            )
            st.plotly_chart(fig_content, use_container_width=True)

    # Support tickets + CSM notes
    tc1, tc2 = st.columns(2)

    with tc1:
        st.markdown("#### 🎫 Support Tickets")
        from renewal_intel.ingest import load_support
        tickets = load_support()
        acct_tickets = tickets[tickets["account_id"] == row["account_id"]].sort_values("created_date", ascending=False)
        if not acct_tickets.empty:
            display_tickets = acct_tickets[["ticket_id", "created_date", "subject", "priority", "status"]].copy()
            display_tickets["created_date"] = display_tickets["created_date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display_tickets, use_container_width=True, hide_index=True, height=250)
        else:
            st.info("No support tickets for this account.")

    with tc2:
        st.markdown("#### 📝 CSM Notes")
        note_text = row.get("csm_note_text", "")
        if note_text:
            st.markdown(f'<div class="detail-section"><p>{note_text}</p></div>', unsafe_allow_html=True)
        else:
            st.info("No CSM notes matched to this account.")

    # NPS
    if row.get("nps_comment"):
        st.markdown("#### 💬 NPS Verbatim")
        st.markdown(f'> "{row["nps_comment"]}" — NPS Score: **{int(row["nps_score"])}**')

    # Risk factors
    factors = row.get("risk_factors", [])
    if factors and isinstance(factors, list):
        st.markdown("#### ⚠️ All Risk Factors")
        for f in factors:
            st.markdown(f'<span class="factor-pill">{f}</span>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: Non-Obvious Insights
# ═══════════════════════════════════════════════════════════════════════════
elif page == "💡 Non-Obvious Insights":
    st.markdown('''
    <div class="main-header">
        <h1>💡 Non-Obvious Insights</h1>
        <p>Patterns that a simple rule-based system would miss</p>
    </div>
    ''', unsafe_allow_html=True)

    if not insights_data:
        st.info("No non-obvious insights detected in the current cohort.")
    else:
        for idx, insight in enumerate(insights_data):
            severity_class = insight["severity"]
            severity_emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(severity_class, "📋")

            st.markdown(f'''
            <div class="insight-card {severity_class}">
                <h4>{insight["title"]}</h4>
                <p>{insight["description"]}</p>
            </div>
            ''', unsafe_allow_html=True)

            # Show affected accounts
            affected = insight["affected_accounts"]
            if affected:
                with st.expander(f"📊 View {len(affected)} affected accounts", expanded=False):
                    st.dataframe(pd.DataFrame(affected), use_container_width=True, hide_index=True)

            st.markdown("")

    # How insights work
    st.markdown("---")
    st.markdown("### 🧠 How These Insights Work")
    st.markdown("""
    These insights are generated by cross-referencing multiple data sources in ways that 
    single-metric dashboards cannot:

    | Insight | Data Sources Used | Why It's Non-Obvious |
    |---------|-------------------|---------------------|
    | **Silent Churn** | NPS + Usage Metrics | NPS looks healthy, but product adoption is declining |
    | **NPS Mismatch** | NPS Score + Verbatim Text | Score contradicts the actual comment — data quality issue |
    | **SDK Compound Risk** | Changelog + Usage SDK + Support Tickets | Three unrelated sources converge to reveal compounding risk |
    | **Changelog Impact** | Changelog + SDK Versions | Platform changes will break specific accounts' implementations |
    | **Champion Risk** | CSM Notes (NLP) | Qualitative relationship signals from unstructured notes |
    """)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: Architecture
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Architecture":
    st.markdown('''
    <div class="main-header">
        <h1>🏗️ System Architecture</h1>
        <p>How the Renewal Risk Intelligence Engine works</p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("### Pipeline Overview")

    # Mermaid-style architecture using text
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                      DATA SOURCES                          │
    │                                                             │
    │  📄 accounts.csv     📊 usage_metrics.csv                   │
    │  🎫 support_tickets  📝 csm_notes.txt                      │
    │  ⭐ nps_responses    📋 changelog.md                        │
    └────────────────────────────┬────────────────────────────────┘
                                 │
                          ┌──────▼──────┐
                          │   INGEST    │ Parse CSVs, split notes,
                          │             │ extract changelog entries
                          └──────┬──────┘
                                 │
                        ┌────────▼────────┐
                        │   RECONCILE     │ Regex ID extraction +
                        │                 │ RapidFuzz name matching
                        │   "BritePath"   │ → BrightPath Solutions
                        │   "Pinacle"     │ → Pinnacle Media Group
                        └────────┬────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   FEATURE ENGINE      │
                     │                       │
                     │  • Usage trends       │
                     │  • Support metrics    │
                     │  • NPS + sentiment    │
                     │  • SDK version risk   │
                     │  • Changelog impact   │
                     └───────────┬───────────┘
                                 │
                      ┌──────────▼──────────┐
                      │    RISK SCORER      │ Weighted composite
                      │    (0-100 score)    │ 8 sub-scores
                      │                     │
                      │  High ≥ 65          │
                      │  Medium: 40-64      │
                      │  Low < 40           │
                      └──────────┬──────────┘
                                 │
               ┌─────────────────▼──────────────────┐
               │      LLM EXPLANATION (Groq)        │
               │                                     │
               │  Model: llama-3.3-70b-versatile     │
               │  • Per-account risk summary         │
               │  • Recommended actions              │
               │  • Data conflict detection          │
               │  • Fallback: deterministic templates │
               └─────────────────┬──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   INSIGHT ENGINE        │
                    │                         │
                    │  🔇 Silent Churn        │
                    │  ⚠️ NPS Mismatch        │
                    │  🔴 SDK Compound Risk   │
                    │  📋 Changelog Impact    │
                    │  👤 Champion Risk       │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────▼───────────────────┐
              │         STREAMLIT DASHBOARD          │
              │                                       │
              │  🏠 Executive Dashboard               │
              │  📋 Risk Scorecard                    │
              │  🔍 Account Deep Dive                 │
              │  💡 Non-Obvious Insights              │
              │  🏗️ Architecture                      │
              └───────────────────────────────────────┘
    ```
    """)

    st.markdown("---")

    st.markdown("### Risk Scoring Formula")
    st.markdown("""
    The composite risk score (0-100) is computed as a weighted sum of 8 sub-scores:

    | Component | Weight | What It Measures |
    |-----------|--------|------------------|
    | Usage Decline | 20 pts | API call + content creation trend (early vs late 3 months) |
    | Support Burden | 15 pts | Open/P1/escalated tickets, recurring issues |
    | NPS Detractor | 15 pts | Survey score 0-10 mapped to risk |
    | CSM Sentiment | 15 pts | Competitor mentions, negative keywords in notes |
    | SDK Deprecated | 10 pts | v3.x customers facing April 30 sunset |
    | Engagement Decay | 10 pts | Active user trend decline |
    | Conflicting Signals | 10 pts | Silent churn pattern (good NPS + bad usage) |
    | Contract Urgency | 5 pts | Days until renewal (sooner = higher) |
    """)

    st.markdown("---")

    st.markdown("### Key Design Decisions")
    st.markdown("""
    1. **Interpretable scoring** — Weighted rules, not a black-box ML model. Every point in the 
       risk score traces back to a specific signal. This is critical for BizOps teams who need 
       to *explain* risk to account managers.

    2. **Fuzzy reconciliation** — CSM notes are messy by design. We use regex for explicit IDs 
       and RapidFuzz (token_sort_ratio) for name matching. Handles "BritePath" → BrightPath, 
       "Pinacle" → Pinnacle, "Thunderbolt Moters" → Thunderbolt Motors.

    3. **LLM as analyst, not oracle** — The Groq LLM receives all computed signals and produces 
       *contextual explanations* and *action recommendations*. It doesn't make the risk decision — 
       the deterministic scorer does. This ensures consistency while leveraging LLM reasoning for 
       narrative quality.

    4. **Changelog as a risk multiplier** — The changelog reveals SDK deprecation deadlines and 
       breaking changes. Cross-referencing `sdk_version` from usage data with changelog entries 
       surfaces compound risks invisible to any single data source.

    5. **Graceful degradation** — The entire system works without an LLM API key using 
       deterministic templates. This ensures the demo always runs.
    """)

    st.markdown("---")
    st.markdown("### Technology Stack")

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown("""
        **Core**
        - Python 3.10+
        - Pandas & NumPy
        - RapidFuzz
        """)
    with tc2:
        st.markdown("""
        **AI / LLM**
        - Groq API
        - LLaMA 3.3 70B
        - JSON structured output
        """)
    with tc3:
        st.markdown("""
        **UI / Viz**
        - Streamlit
        - Plotly
        - Custom CSS
        """)
