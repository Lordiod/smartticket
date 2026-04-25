"""
SmartTicket — Interactive Dashboard
Streamlit app for exploring data, preprocessing, and voting ensemble results.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from smartticket_core import (
    load_raw_data,
    run_full_pipeline,
    train_all_models,
    predict_single_ticket,
    clean_text,
)

# ══════════════════════════════════════════════════════════════
# Page Config & Global Styles
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SmartTicket",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Root variables ── */
    :root {
        --accent: #6C63FF;
        --accent-light: #8B83FF;
        --surface: #0E1117;
        --card: #1A1D29;
        --card-border: #2A2D3A;
        --success: #00D68F;
        --warning: #FFAA00;
        --danger: #FF3D71;
        --text-primary: #E4E6EB;
        --text-secondary: #8B8D97;
    }

    /* ── Hide default streamlit branding ── */
    /* Note: do not hide `header` — Streamlit puts the sidebar expand control there;
       hiding the whole header makes a collapsed sidebar impossible to reopen. */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12141D 0%, #1A1D29 100%);
        border-right: 1px solid var(--card-border);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        padding: 0.4rem 0;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1D29 0%, #22253A 100%);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* ── Section headers ── */
    .section-header {
        background: linear-gradient(90deg, var(--accent) 0%, #4834D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 1.5rem 0 0.8rem 0;
        letter-spacing: -0.02em;
    }

    /* ── Glass card ── */
    .glass-card {
        background: linear-gradient(135deg, rgba(26,29,41,0.9) 0%, rgba(34,37,58,0.9) 100%);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }

    /* ── Voting result badge ── */
    .vote-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
    }
    .vote-hard { background: rgba(108,99,255,0.2); color: #8B83FF; border: 1px solid rgba(108,99,255,0.3); }
    .vote-soft { background: rgba(0,214,143,0.2); color: #00D68F; border: 1px solid rgba(0,214,143,0.3); }
    .vote-weighted { background: rgba(255,170,0,0.2); color: #FFAA00; border: 1px solid rgba(255,170,0,0.3); }

    /* ── Accent divider ── */
    .accent-divider {
        height: 3px;
        background: linear-gradient(90deg, var(--accent), transparent);
        border: none;
        margin: 1rem 0;
        border-radius: 2px;
    }

    /* ── Data tables ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }

    /* ── Top hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(108,99,255,0.2);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .hero-subtitle {
        color: #8B8D97;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* ── Prediction card ── */
    .prediction-result {
        background: linear-gradient(135deg, #1a2332 0%, #1a1d29 100%);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .prediction-value {
        font-size: 1.5rem;
        font-weight: 800;
        text-transform: capitalize;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Caching — load data & train models once
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cached_pipeline():
    return run_full_pipeline()


@st.cache_data(show_spinner=False)
def cached_models(_X_train, _X_val, _y_train, _y_val, _le_dept):
    return train_all_models(_X_train, _X_val, _y_train, _y_val, _le_dept)


# ══════════════════════════════════════════════════════════════
# Plotly Theme
# ══════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#E4E6EB"),
    margin=dict(l=40, r=40, t=50, b=40),
)

PALETTE = ["#6C63FF", "#00D68F", "#FF6B6B", "#FFAA00", "#38BDF8", "#F472B6",
           "#A78BFA", "#34D399", "#FB923C", "#818CF8"]


# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎫 SmartTicket")
    st.markdown('<div class="accent-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 Data Explorer",
            "🔬 Preprocessing",
            "🗳️ Voting Ensemble",
            "⚔️ Model Arena",
            "🔮 Live Predictor",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Built with Streamlit + Plotly")
    st.caption("SmartTicket ML Pipeline")


# ══════════════════════════════════════════════════════════════
# Load everything
# ══════════════════════════════════════════════════════════════

with st.spinner("⚡ Loading pipeline & training models..."):
    pipeline = cached_pipeline()
    models = cached_models(
        pipeline["X_train"], pipeline["X_val"],
        pipeline["y_train"], pipeline["y_val"],
        pipeline["le_dept"],
    )

df_raw = pipeline["df_raw"]
df = pipeline["df"]
classes = models["classes"]


# ══════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">SmartTicket Dashboard</p>
        <p class="hero-subtitle">AI-powered support ticket classification — routing tickets to the right department using ensemble ML</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    best_model = max(
        [(k, v["accuracy"]) for k, v in models.items() if isinstance(v, dict) and "accuracy" in v],
        key=lambda x: x[1],
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tickets", f"{len(df_raw):,}")
    c2.metric("After Cleaning", f"{len(df):,}")
    c3.metric("Departments", len(classes))
    c4.metric("Best Accuracy", f"{best_model[1]:.1%}")

    st.markdown("")

    # Quick comparison chart
    st.markdown('<p class="section-header">Model Performance at a Glance</p>', unsafe_allow_html=True)

    model_names = ["Baseline KNN", "Hard Voting", "Soft Voting", "Weighted Soft"]
    model_keys = ["baseline", "hard_voting", "soft_voting", "weighted_voting"]
    accs = [models[k]["accuracy"] for k in model_keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=model_names, y=accs,
        marker=dict(
            color=accs,
            colorscale=[[0, "#FF6B6B"], [0.5, "#FFAA00"], [1, "#00D68F"]],
            line=dict(width=0),
            cornerradius=6,
        ),
        text=[f"{a:.1%}" for a in accs],
        textposition="outside",
        textfont=dict(size=16, color="#E4E6EB", family="Inter, sans-serif"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        yaxis=dict(range=[0, max(accs) * 1.18], title="Accuracy", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title=""),
        height=420,
        showlegend=False,
        title=dict(text="Accuracy Comparison", font=dict(size=18)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Individual model breakdown
    st.markdown('<p class="section-header">Individual Classifier Accuracy</p>', unsafe_allow_html=True)

    indiv = models["individual_accuracies"]
    cols = st.columns(len(indiv))
    colors = ["#6C63FF", "#00D68F", "#38BDF8", "#FFAA00", "#F472B6"]
    for i, (name, acc) in enumerate(indiv.items()):
        with cols[i]:
            st.metric(name, f"{acc:.1%}")

    # Department distribution
    st.markdown('<p class="section-header">Department Distribution</p>', unsafe_allow_html=True)

    dept_counts = df["department"].value_counts().reset_index()
    dept_counts.columns = ["Department", "Count"]
    fig_dept = px.bar(
        dept_counts, x="Department", y="Count",
        color="Department", color_discrete_sequence=PALETTE,
    )
    fig_dept.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False,
                           xaxis=dict(title=""), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
    fig_dept.update_traces(marker=dict(cornerradius=6))
    st.plotly_chart(fig_dept, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Data Explorer
# ══════════════════════════════════════════════════════════════

elif page == "📊 Data Explorer":
    st.markdown('<p class="section-header">Data Explorer</p>', unsafe_allow_html=True)
    st.markdown("Browse the raw ticket data and explore distributions interactively.")

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📈 Distributions", "🔗 Correlations"])

    with tab1:
        col_filter, col_dept, col_prio = st.columns(3)
        with col_dept:
            dept_filter = st.multiselect("Department", options=sorted(df_raw["department"].dropna().unique()))
        with col_prio:
            prio_filter = st.multiselect("Priority", options=sorted(df_raw["priority"].dropna().unique()))
        with col_filter:
            search = st.text_input("Search ticket text", placeholder="Type to filter...")

        filtered = df_raw.copy()
        if dept_filter:
            filtered = filtered[filtered["department"].isin(dept_filter)]
        if prio_filter:
            filtered = filtered[filtered["priority"].isin(prio_filter)]
        if search:
            filtered = filtered[filtered["ticket_text"].fillna("").str.contains(search, case=False)]

        st.dataframe(
            filtered[["ticket_id", "ticket_text", "department", "priority", "channel", "product_category"]],
            use_container_width=True,
            height=450,
        )
        st.caption(f"Showing {len(filtered):,} of {len(df_raw):,} tickets")

    with tab2:
        d1, d2 = st.columns(2)
        with d1:
            fig_ch = px.pie(
                df_raw, names="channel", title="Channel Distribution",
                color_discrete_sequence=PALETTE, hole=0.45,
            )
            fig_ch.update_layout(**PLOTLY_LAYOUT, height=380)
            fig_ch.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_ch, use_container_width=True)

        with d2:
            fig_prod = px.pie(
                df_raw, names="product_category", title="Product Category",
                color_discrete_sequence=PALETTE[3:], hole=0.45,
            )
            fig_prod.update_layout(**PLOTLY_LAYOUT, height=380)
            fig_prod.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_prod, use_container_width=True)

        # Priority by department
        cross = pd.crosstab(df_raw["department"], df_raw["priority"])
        fig_cross = px.bar(
            cross.reset_index().melt(id_vars="department"),
            x="department", y="value", color="priority",
            barmode="group", color_discrete_sequence=PALETTE,
            title="Priority by Department",
        )
        fig_cross.update_layout(**PLOTLY_LAYOUT, height=400,
                                yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.05)"))
        fig_cross.update_traces(marker=dict(cornerradius=4))
        st.plotly_chart(fig_cross, use_container_width=True)

    with tab3:
        numeric_for_corr = ["total_spent", "total_orders", "returns_count", "avg_order_value",
                            "previous_tickets", "avg_response_satisfaction", "sentiment_score",
                            "response_time_hours", "num_replies"]
        numeric_for_corr = [c for c in numeric_for_corr if c in df.columns]
        corr = df[numeric_for_corr].corr()

        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            aspect="auto", title="Feature Correlation Matrix",
        )
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=550)
        st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Preprocessing
# ══════════════════════════════════════════════════════════════

elif page == "🔬 Preprocessing":
    st.markdown('<p class="section-header">Preprocessing Pipeline</p>', unsafe_allow_html=True)
    st.markdown("See how raw tickets are transformed step-by-step before feeding into the models.")

    # Pipeline diagram
    st.markdown("""
    <div class="glass-card">
        <div style="display:flex; align-items:center; justify-content:center; flex-wrap:wrap; gap:0.5rem;">
            <span class="vote-badge vote-hard">Raw Text</span>
            <span style="color:#8B8D97;">→</span>
            <span class="vote-badge vote-soft">Clean</span>
            <span style="color:#8B8D97;">→</span>
            <span class="vote-badge vote-weighted">Stopwords</span>
            <span style="color:#8B8D97;">→</span>
            <span class="vote-badge vote-hard">Lemmatize</span>
            <span style="color:#8B8D97;">→</span>
            <span class="vote-badge vote-soft">TF-IDF</span>
            <span style="color:#8B8D97;">→</span>
            <span class="vote-badge vote-weighted">Scale</span>
            <span style="color:#8B8D97;">→</span>
            <span class="vote-badge vote-hard">Feature Matrix</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Live text cleaning demo
    st.markdown("#### 🧹 Try Text Cleaning")
    sample_raw = df_raw["ticket_text"].dropna().iloc[0]
    user_text = st.text_area("Paste or edit a ticket to see cleaning in action:", value=sample_raw, height=100)

    if user_text:
        cleaned = clean_text(user_text)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Raw Input**")
            st.code(user_text, language=None)
        with c2:
            st.markdown("**After Cleaning**")
            st.code(cleaned, language=None)

    st.markdown("---")

    # Data quality metrics
    st.markdown("#### 📊 Data Quality Summary")
    q1, q2, q3, q4 = st.columns(4)
    missing_text = df_raw["ticket_text"].isna().sum()
    q1.metric("Missing Texts", missing_text, help="Ticket texts that were NULL")
    q2.metric("Duplicates Removed", pipeline["n_dupes"])
    q3.metric("Features (TF-IDF)", 500)
    total_feats = pipeline["X_train"].shape[1]
    q4.metric("Total Features", total_feats)

    # Before / after samples
    st.markdown("#### 🔍 Before & After Samples")
    n_samples = st.slider("Samples to show", 1, 10, 5)
    samples = df_raw[["ticket_id", "ticket_text"]].dropna().head(n_samples).copy()
    samples["cleaned"] = samples["ticket_text"].apply(clean_text)
    samples = samples.rename(columns={"ticket_text": "original"})
    st.dataframe(samples, use_container_width=True, height=min(n_samples * 60 + 50, 500))

    # Feature space breakdown
    st.markdown("#### 🧩 Feature Space Breakdown")
    feat_breakdown = pd.DataFrame({
        "Feature Group": ["TF-IDF (text)", "Numeric (scaled)", "One-Hot Encoded"],
        "Count": [500, len(pipeline["all_numeric"]), len(pipeline["ohe_cols"])],
    })
    fig_feat = px.bar(
        feat_breakdown, x="Feature Group", y="Count",
        color="Feature Group", color_discrete_sequence=PALETTE[:3],
        text="Count",
    )
    fig_feat.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False,
                           yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
    fig_feat.update_traces(textposition="outside", marker=dict(cornerradius=6))
    st.plotly_chart(fig_feat, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Voting Ensemble
# ══════════════════════════════════════════════════════════════

elif page == "🗳️ Voting Ensemble":
    st.markdown('<p class="section-header">Voting Ensemble Deep Dive</p>', unsafe_allow_html=True)

    # Explain voting strategies
    st.markdown("""
    <div class="glass-card">
        <div style="display:flex; gap:1.5rem; flex-wrap:wrap; justify-content:center;">
            <div style="flex:1; min-width:200px; text-align:center;">
                <span class="vote-badge vote-hard">Hard Voting</span>
                <p style="color:#8B8D97; font-size:0.85rem; margin-top:0.6rem;">
                    Each model casts one vote.<br/>Majority class wins.
                </p>
            </div>
            <div style="flex:1; min-width:200px; text-align:center;">
                <span class="vote-badge vote-soft">Soft Voting</span>
                <p style="color:#8B8D97; font-size:0.85rem; margin-top:0.6rem;">
                    Average predicted probabilities.<br/>Highest average probability wins.
                </p>
            </div>
            <div style="flex:1; min-width:200px; text-align:center;">
                <span class="vote-badge vote-weighted">Weighted Soft</span>
                <p style="color:#8B8D97; font-size:0.85rem; margin-top:0.6rem;">
                    Like soft, but better models<br/>get more influence via weights.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Accuracy cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Hard Voting", f"{models['hard_voting']['accuracy']:.1%}")
    c2.metric("Soft Voting", f"{models['soft_voting']['accuracy']:.1%}")
    c3.metric("Weighted Soft", f"{models['weighted_voting']['accuracy']:.1%}")

    st.markdown("---")

    # Individual model contributions
    st.markdown("#### 🏗️ Individual Model Contributions")

    indiv = models["individual_accuracies"]
    weights = models["weights"]

    indiv_df = pd.DataFrame({
        "Model": list(indiv.keys()),
        "Accuracy": list(indiv.values()),
        "Weight": [weights[n] for n in indiv.keys()],
    })
    indiv_df["Weight (normalized)"] = indiv_df["Weight"] / indiv_df["Weight"].sum()

    fig_indiv = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Individual Accuracy", "Voting Weights"),
        specs=[[{"type": "bar"}, {"type": "pie"}]],
    )

    fig_indiv.add_trace(
        go.Bar(
            x=indiv_df["Model"], y=indiv_df["Accuracy"],
            marker=dict(color=PALETTE[:len(indiv_df)], cornerradius=6),
            text=[f"{a:.1%}" for a in indiv_df["Accuracy"]],
            textposition="outside",
        ),
        row=1, col=1,
    )

    fig_indiv.add_trace(
        go.Pie(
            labels=indiv_df["Model"],
            values=indiv_df["Weight (normalized)"],
            marker=dict(colors=PALETTE[:len(indiv_df)]),
            textinfo="label+percent",
            hole=0.4,
        ),
        row=1, col=2,
    )

    fig_indiv.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False)
    fig_indiv.update_yaxes(range=[0, max(indiv_df["Accuracy"]) * 1.15], gridcolor="rgba(255,255,255,0.05)", row=1, col=1)
    st.plotly_chart(fig_indiv, use_container_width=True)

    # Confusion matrices
    st.markdown("#### 🔲 Confusion Matrices")
    cm_tab1, cm_tab2, cm_tab3 = st.tabs(["Hard Voting", "Soft Voting", "Weighted Soft"])

    for tab, key, title in [
        (cm_tab1, "hard_voting", "Hard Voting"),
        (cm_tab2, "soft_voting", "Soft Voting"),
        (cm_tab3, "weighted_voting", "Weighted Soft Voting"),
    ]:
        with tab:
            cm = models[key]["confusion"]
            fig_cm = px.imshow(
                cm, x=classes, y=classes,
                color_continuous_scale="Blues",
                text_auto=True, aspect="auto",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                title=f"{title} — Confusion Matrix",
            )
            fig_cm.update_layout(**PLOTLY_LAYOUT, height=450)
            st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class F1 scores
    st.markdown("#### 📊 Per-Class F1 Scores")

    f1_data = []
    for key, label in [("hard_voting", "Hard"), ("soft_voting", "Soft"), ("weighted_voting", "Weighted Soft")]:
        report = models[key]["report"]
        for cls in classes:
            f1_data.append({"Class": cls, "Strategy": label, "F1-Score": report[cls]["f1-score"]})

    f1_df = pd.DataFrame(f1_data)
    fig_f1 = px.bar(
        f1_df, x="Class", y="F1-Score", color="Strategy",
        barmode="group", color_discrete_sequence=["#6C63FF", "#00D68F", "#FFAA00"],
        text_auto=".2f",
    )
    fig_f1.update_layout(**PLOTLY_LAYOUT, height=420,
                         yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.05)"),
                         legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
    fig_f1.update_traces(marker=dict(cornerradius=4), textposition="outside")
    st.plotly_chart(fig_f1, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Model Arena
# ══════════════════════════════════════════════════════════════

elif page == "⚔️ Model Arena":
    st.markdown('<p class="section-header">Model Arena</p>', unsafe_allow_html=True)
    st.markdown("Head-to-head comparison of all approaches.")

    # Full comparison table
    all_results = {
        "Baseline KNN": models["baseline"],
        "Hard Voting": models["hard_voting"],
        "Soft Voting": models["soft_voting"],
        "Weighted Soft Voting": models["weighted_voting"],
    }

    comparison_data = []
    for name, data in all_results.items():
        report = data["report"]
        comparison_data.append({
            "Model": name,
            "Accuracy": data["accuracy"],
            "Macro F1": report["macro avg"]["f1-score"],
            "Weighted F1": report["weighted avg"]["f1-score"],
            "Macro Precision": report["macro avg"]["precision"],
            "Macro Recall": report["macro avg"]["recall"],
        })

    comp_df = pd.DataFrame(comparison_data)

    # Highlight best
    st.dataframe(
        comp_df.style.highlight_max(
            subset=["Accuracy", "Macro F1", "Weighted F1", "Macro Precision", "Macro Recall"],
            color="rgba(0, 214, 143, 0.3)",
        ).format({
            "Accuracy": "{:.2%}",
            "Macro F1": "{:.4f}",
            "Weighted F1": "{:.4f}",
            "Macro Precision": "{:.4f}",
            "Macro Recall": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # Radar chart
    st.markdown("#### 🎯 Multi-Metric Radar")

    radar_metrics = ["Accuracy", "Macro F1", "Weighted F1", "Macro Precision", "Macro Recall"]

    fig_radar = go.Figure()
    colors_radar = ["#FF6B6B", "#6C63FF", "#00D68F", "#FFAA00"]
    for i, row in comp_df.iterrows():
        values = [row[m] for m in radar_metrics]
        values.append(values[0])  # close the polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_metrics + [radar_metrics[0]],
            fill="toself",
            fillcolor=f"rgba({int(colors_radar[i][1:3],16)},{int(colors_radar[i][3:5],16)},{int(colors_radar[i][5:7],16)},0.1)",
            line=dict(color=colors_radar[i], width=2),
            name=row["Model"],
        ))

    fig_radar.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ),
        height=500,
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Ensemble gain analysis
    st.markdown("#### 📈 Ensemble Gain Over Baseline")
    baseline_acc = models["baseline"]["accuracy"]
    gains = {name: data["accuracy"] - baseline_acc for name, data in all_results.items() if name != "Baseline KNN"}

    fig_gain = go.Figure()
    fig_gain.add_trace(go.Bar(
        x=list(gains.keys()),
        y=[g * 100 for g in gains.values()],
        marker=dict(
            color=["#6C63FF", "#00D68F", "#FFAA00"],
            cornerradius=6,
        ),
        text=[f"+{g:.2f}pp" for g in [g * 100 for g in gains.values()]],
        textposition="outside",
        textfont=dict(size=14),
    ))
    fig_gain.update_layout(
        **PLOTLY_LAYOUT, height=380,
        yaxis=dict(title="Accuracy Gain (percentage points)", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title=""),
        title=dict(text=f"Improvement over Baseline ({baseline_acc:.1%})", font=dict(size=16)),
    )
    fig_gain.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    st.plotly_chart(fig_gain, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Live Predictor
# ══════════════════════════════════════════════════════════════

elif page == "🔮 Live Predictor":
    st.markdown('<p class="section-header">Live Ticket Classifier</p>', unsafe_allow_html=True)
    st.markdown("Type a support ticket and watch all models classify it in real-time.")

    example_tickets = {
        "Select an example...": "",
        "Billing issue": "I was charged twice for my subscription this month. Please refund the duplicate payment immediately. My account shows $49.99 twice on March 15.",
        "Technical problem": "The software keeps crashing when I try to export data to CSV. I get an error code 500 and the application freezes completely. I'm running version 3.2 on Windows 11.",
        "Shipping delay": "My order #45231 was supposed to arrive 5 days ago but tracking still shows it in transit. I need this package urgently for a business presentation.",
        "Account access": "I can't log into my account. I've tried resetting my password three times but I never receive the reset email. My username is john.doe@example.com.",
        "Return request": "I received a damaged product and would like to return it for a full refund. The electronics item arrived with a cracked screen and doesn't turn on.",
        "General inquiry": "I'm interested in upgrading my current plan. Can you tell me what features are included in the premium tier and how much it costs per month?",
    }

    selected_example = st.selectbox("Quick examples", options=list(example_tickets.keys()))

    ticket_text = st.text_area(
        "Enter ticket text",
        value=example_tickets[selected_example],
        height=120,
        placeholder="Describe your support issue here...",
    )

    if st.button("🚀 Classify Ticket", type="primary", use_container_width=True):
        if ticket_text.strip():
            with st.spinner("Running through all models..."):
                results = predict_single_ticket(ticket_text, pipeline, models)

            st.markdown("")

            # Show cleaned text
            with st.expander("🧹 Preprocessed text", expanded=False):
                st.code(clean_text(ticket_text), language=None)

            # Results grid
            st.markdown("#### Predictions")

            display_names = {
                "baseline": ("Baseline KNN", "#FF6B6B"),
                "hard_voting": ("Hard Voting", "#6C63FF"),
                "soft_voting": ("Soft Voting", "#00D68F"),
                "weighted_voting": ("Weighted Soft", "#FFAA00"),
            }

            cols = st.columns(4)
            for i, (key, dept) in enumerate(results.items()):
                name, color = display_names[key]
                with cols[i]:
                    st.markdown(f"""
                    <div class="prediction-result">
                        <div class="prediction-label">{name}</div>
                        <div class="prediction-value" style="color:{color};">{dept}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Agreement analysis
            st.markdown("")
            unique_preds = set(results.values())
            if len(unique_preds) == 1:
                st.success(f"✅ **Unanimous agreement** — All models predict **{list(unique_preds)[0]}**")
            else:
                from collections import Counter
                vote_counts = Counter(results.values())
                majority = vote_counts.most_common(1)[0]
                st.warning(
                    f"⚠️ **Split vote** — {len(unique_preds)} different predictions. "
                    f"Majority: **{majority[0]}** ({majority[1]}/{len(results)} models)"
                )

            # Voting breakdown
            st.markdown("#### 🗳️ Vote Breakdown")
            from collections import Counter
            vote_counts = Counter(results.values())
            vote_df = pd.DataFrame(
                [{"Department": dept, "Votes": count} for dept, count in vote_counts.most_common()]
            )
            fig_votes = px.bar(
                vote_df, x="Department", y="Votes",
                color="Department", color_discrete_sequence=PALETTE,
                text="Votes",
            )
            fig_votes.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False,
                                    yaxis=dict(dtick=1, gridcolor="rgba(255,255,255,0.05)"))
            fig_votes.update_traces(textposition="outside", marker=dict(cornerradius=6))
            st.plotly_chart(fig_votes, use_container_width=True)
        else:
            st.warning("Please enter some ticket text first.")
