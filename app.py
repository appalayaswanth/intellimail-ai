"""IntelliMail AI — AI-Powered Email Marketing Automation System
Refined editorial luxury interface with warm ivory canvas, charcoal typography, and muted olive accents.
"""

import os
import uuid
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st

# Local modules
from config import (
    APP_TITLE,
    APP_SUBTITLE,
    BRAND_COLORS,
    GEMINI_API_KEY,
    BREVO_API_KEY,
    AVAILABLE_TONES,
    CUSTOMER_INTERESTS,
    DATA_DIR
)
from database import (
    init_db,
    upsert_customers_from_df,
    get_all_customers_df,
    get_customer_count,
    update_customer_segments,
    update_customer_engagement_scores,
    unsubscribe_customer,
    resubscribe_customer,
    save_campaign,
    get_all_campaigns,
    get_campaign_logs,
    save_template,
    get_templates,
    get_dashboard_kpis
)
from modules.ai_generator import AIEmailGenerator
from modules.segmentation import (
    generate_synthetic_customers,
    perform_rfm_clustering,
    plot_rfm_scatter_3d,
    plot_segment_distribution,
    plot_segment_monetary_comparison
)
from modules.engagement_model import (
    train_engagement_model,
    load_saved_model,
    predict_customer_engagement,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)
from modules.email_automation import (
    EmailAutomationService,
    plot_campaign_comparison,
    plot_campaign_rates_pie
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="IntelliMail AI — Marketing Automation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Editorial Luxury CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..600;1,6..72,400..500&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    :root {
        --canvas-bg: #FAF8F5;
        --surface-bg: #FFFFFF;
        --surface-subtle: #F4EFEA;
        --border-subtle: #E8E3D9;
        --border-refined: #DCD6CA;
        --ink-primary: #191C18;
        --ink-secondary: #3B4238;
        --ink-muted: #737B6E;
        --olive-primary: #4E5F47;
        --olive-hover: #3D4C37;
        --olive-subtle: #EDF2EA;
        --ochre: #8F754E;
        --terracotta: #935353;
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: var(--ink-secondary);
        background-color: var(--canvas-bg);
    }
    
    .stApp {
        background-color: var(--canvas-bg);
    }

    /* Editorial Masthead Header */
    .editorial-masthead {
        border-bottom: 1px solid var(--border-subtle);
        padding: 10px 0 28px 0;
        margin-bottom: 32px;
    }
    .masthead-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .masthead-kicker {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: var(--ink-muted);
    }
    .masthead-status {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: var(--olive-primary);
        background-color: var(--olive-subtle);
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #D7E2D3;
    }
    .masthead-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 40px;
        font-weight: 400;
        letter-spacing: -0.02em;
        color: var(--ink-primary);
        margin: 0;
        line-height: 1.15;
    }
    .masthead-deck {
        font-size: 15px;
        color: var(--ink-muted);
        margin: 8px 0 0 0;
        font-weight: 400;
        max-width: 720px;
        line-height: 1.5;
    }

    /* Page Intro Section */
    .page-intro {
        margin-bottom: 28px;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .page-kicker {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: var(--olive-primary);
        margin-bottom: 6px;
    }
    .page-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 30px;
        font-weight: 400;
        color: var(--ink-primary);
        margin: 0;
        letter-spacing: -0.015em;
    }
    .page-deck {
        font-size: 14px;
        color: var(--ink-muted);
        margin-top: 6px;
        line-height: 1.5;
    }

    /* Minimal Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F4F0E8 !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] .sidebar-brand {
        padding: 12px 4px 20px 4px;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 22px;
    }
    .sidebar-brand-kicker {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: var(--ink-muted);
    }
    .sidebar-brand-name {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 24px;
        font-weight: 400;
        color: var(--ink-primary);
        letter-spacing: -0.01em;
        margin-top: 2px;
    }
    .sidebar-brand-desc {
        font-size: 11.5px;
        color: var(--ink-muted);
        margin-top: 2px;
    }
    
    /* Clean Editorial Radio Navigation */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 2px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        display: flex;
        align-items: center;
        padding: 9px 12px;
        border-radius: 6px;
        font-size: 13.5px;
        font-weight: 500;
        color: var(--ink-secondary);
        background-color: transparent;
        border-left: 2px solid transparent;
        transition: all 0.18s ease;
        cursor: pointer;
        margin-bottom: 2px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #EAE4D8;
        color: var(--ink-primary);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background-color: #E5DFD1;
        color: var(--ink-primary);
        border-left: 2px solid var(--olive-primary);
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        display: none;
    }

    /* Sidebar Status Badges */
    .sidebar-meta-box {
        background-color: #FAF8F5;
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 14px;
        margin-top: 20px;
    }
    .sidebar-meta-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: var(--ink-muted);
        margin-bottom: 8px;
    }
    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0;
        font-size: 12.5px;
    }

    /* Editorial Metric Cards */
    .metric-card {
        background: var(--surface-bg);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 22px 20px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .metric-card:hover {
        border-color: var(--border-refined);
        transform: translateY(-1px);
    }
    .metric-kicker {
        font-size: 10.5px;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: var(--ink-muted);
    }
    .metric-val {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 32px;
        font-weight: 400;
        color: var(--ink-primary);
        margin: 6px 0 4px 0;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .metric-caption {
        font-size: 12px;
        color: var(--olive-primary);
        font-weight: 500;
    }

    /* Refined Buttons */
    .stButton button[kind="primary"],
    .stButton > button[kind="primary"] {
        background-color: var(--olive-primary) !important;
        border: 1px solid var(--olive-primary) !important;
        color: #FAF8F5 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        letter-spacing: 0.2px !important;
        padding: 0.55rem 1.2rem !important;
        transition: background-color 0.2s ease !important;
    }
    .stButton button[kind="primary"]:hover,
    .stButton > button[kind="primary"]:hover {
        background-color: var(--olive-hover) !important;
        border-color: var(--olive-hover) !important;
        color: #FFFFFF !important;
    }
    .stButton button:not([kind="primary"]),
    .stButton > button:not([kind="primary"]) {
        background-color: #FAF8F5 !important;
        border: 1px solid var(--border-refined) !important;
        color: var(--ink-secondary) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    .stButton button:not([kind="primary"]):hover,
    .stButton > button:not([kind="primary"]):hover {
        background-color: #F1ECE3 !important;
        border-color: #CEC7BA !important;
        color: var(--ink-primary) !important;
    }

    /* Clean Form Elements */
    .stTextInput > div > div,
    .stTextArea > div > div,
    .stSelectbox > div > div {
        background-color: var(--surface-bg) !important;
        border: 1px solid var(--border-refined) !important;
        border-radius: 6px !important;
        color: var(--ink-primary) !important;
    }
    .stTextInput > div > div:focus-within,
    .stTextArea > div > div:focus-within,
    .stSelectbox > div > div:focus-within {
        border-color: var(--olive-primary) !important;
        box-shadow: 0 0 0 1px var(--olive-primary) !important;
    }

    /* Editorial Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 13.5px;
        font-weight: 500;
        color: var(--ink-muted);
        padding: 8px 16px;
        border-radius: 4px 4px 0 0;
        background-color: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        color: var(--ink-primary) !important;
        border-bottom: 2px solid var(--olive-primary) !important;
        font-weight: 600 !important;
    }

    /* Minimal Segment Badges */
    .pill-champion {
        background-color: #EDF2EB;
        color: #354D39;
        border: 1px solid #D4E0D1;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11.5px;
        font-weight: 600;
    }
    .pill-loyal {
        background-color: #EBF0ED;
        color: #2E3F34;
        border: 1px solid #CFDDD4;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11.5px;
        font-weight: 600;
    }
    .pill-promising {
        background-color: #F7F2EB;
        color: #6E5336;
        border: 1px solid #E8DDD0;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11.5px;
        font-weight: 600;
    }
    .pill-atrisk {
        background-color: #F8ECEC;
        color: #723B3B;
        border: 1px solid #E9D2D2;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11.5px;
        font-weight: 600;
    }
    
    /* Content Panels */
    .editorial-panel {
        background: var(--surface-bg);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 24px;
    }
    .editorial-panel-header {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 20px;
        font-weight: 400;
        color: var(--ink-primary);
        margin: 0 0 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Initialize Database and State
# ---------------------------------------------------------
init_db()

# Handle unsubscribe link simulation if url parameter present
query_params = st.query_params
if "unsubscribe" in query_params:
    unsub_email = query_params["unsubscribe"]
    success = unsubscribe_customer(unsub_email)
    if success:
        st.success(f"Consent preference recorded: {unsub_email} has been unsubscribed from all future marketing campaigns.")
    st.query_params.clear()

if "current_generated_email" not in st.session_state:
    st.session_state.current_generated_email = {
        "subject": "Empower Your Team with AI-Driven Workflow Automation",
        "body": "Dear Valued Customer,\n\nWe are pleased to introduce our latest automation toolkit, engineered to streamline operations and accelerate project velocity.\n\nKey advantages:\n- Real-time predictive analytics\n- Automated campaign segmentation\n- High deliverability architecture\n\nReady to transform your operations?\n\n[Schedule a Technical Consultation]\n\nWarm regards,\nThe IntelliMail Team",
        "tone": "Professional",
        "mode": "Active AI Generator"
    }

if "last_training_results" not in st.session_state:
    st.session_state.last_training_results = None

# Initialize customer database if empty
customer_count = get_customer_count()
if customer_count == 0:
    bootstrap_df = generate_synthetic_customers(n=3000, seed=42)
    clustered_df, _, _, _ = perform_rfm_clustering(bootstrap_df, n_clusters=4, save_models=True)
    initial_probs = predict_customer_engagement(clustered_df, pipeline=None)
    clustered_df["engagement_score"] = initial_probs
    upsert_customers_from_df(clustered_df)
    customer_count = len(clustered_df)

# ---------------------------------------------------------
# Minimal Left-Side Navigation & Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-kicker">COMMERCIAL SUITE</div>
        <div class="sidebar-brand-name">IntelliMail AI</div>
        <div class="sidebar-brand-desc">Marketing Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    nav_options = [
        "Overview",
        "AI Email Generator",
        "Customer Segmentation",
        "Engagement Prediction",
        "Email Campaigns & Analytics"
    ]
    
    selected_page = st.radio(
        "Navigation",
        options=nav_options,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Minimal System Status Panel
    kpi_quick = get_dashboard_kpis()
    st.markdown(f"""
    <div class="sidebar-meta-box">
        <div class="sidebar-meta-label">SYSTEM TELEMETRY</div>
        <div class="status-row">
            <span>AI Studio</span>
            <span class="pill-champion">Active</span>
        </div>
        <div class="status-row">
            <span>Email Service</span>
            <span class="pill-champion">Connected</span>
        </div>
        <div class="status-row">
            <span>ML Pipeline</span>
            <span class="pill-champion">Trained</span>
        </div>
        <hr style="border: 0; border-top: 1px solid #E8E3D9; margin: 10px 0;">
        <div class="sidebar-meta-label">DATABASE METRICS</div>
        <div class="status-row">
            <span>Total Audience</span>
            <span style="font-weight: 600; color: #191C18;">{customer_count:,}</span>
        </div>
        <div class="status-row">
            <span>Campaigns</span>
            <span style="font-weight: 600; color: #191C18;">{kpi_quick['total_campaigns']}</span>
        </div>
        <div class="status-row">
            <span>Subscribed</span>
            <span style="font-weight: 600; color: #191C18;">{kpi_quick['subscribed_customers']:,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Top Editorial Masthead Header
# ---------------------------------------------------------
st.markdown(f"""
<div class="editorial-masthead">
    <div class="masthead-meta">
        <span class="masthead-kicker">MARKETING INTELLIGENCE &bull; ENTERPRISE SUITE</span>
        <span class="masthead-status">SYSTEMS OPERATIONAL</span>
    </div>
    <h1 class="masthead-title">{APP_TITLE}</h1>
    <p class="masthead-deck">{APP_SUBTITLE} &mdash; A unified environment for behavioral segmentation, predictive propensity modeling, and automated campaign orchestration.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# PAGE 1: OVERVIEW
# =========================================================
if selected_page == "Overview":
    st.markdown("""
    <div class="page-intro">
        <div class="page-kicker">EXECUTIVE DASHBOARD</div>
        <h2 class="page-title">Platform Performance & Overview</h2>
        <p class="page-deck">Real-time delivery telemetry, active audience health, and campaign records.</p>
    </div>
    """, unsafe_allow_html=True)

    kpis = get_dashboard_kpis()

    # Metric Cards Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-kicker">Audience Pool</div>
            <div class="metric-val">{kpis['total_customers']:,}</div>
            <div class="metric-caption">{kpis['subscribed_customers']:,} Consented</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-kicker">Campaigns</div>
            <div class="metric-val">{kpis['total_campaigns']}</div>
            <div class="metric-caption">{kpis['total_emails_sent']:,} Dispatched</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-kicker">Delivery Rate</div>
            <div class="metric-val">{(kpis['delivered_emails']/(kpis['total_emails_sent'] or 1)*100):.1f}%</div>
            <div class="metric-caption">{kpis['delivered_emails']:,} Confirmed</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-kicker">Average Open</div>
            <div class="metric-val">{kpis['open_rate']}%</div>
            <div class="metric-caption">{kpis['opened_emails']:,} Total Opens</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-kicker">Average Click</div>
            <div class="metric-val">{kpis['click_rate']}%</div>
            <div class="metric-caption">{kpis['clicked_emails']:,} Link Clicks</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_arch, col_actions = st.columns([2, 1])
    with col_arch:
        st.markdown("""
        <div class="editorial-panel">
            <h3 class="editorial-panel-header">System Architecture & Pipeline Flow</h3>
            <p style="font-size: 14px; line-height: 1.6; color: #3B4238; margin: 0 0 14px 0;">
                IntelliMail AI executes an integrated four-stage workflow engineered for commercial-grade marketing operations:
            </p>
            <ul style="font-size: 13.5px; line-height: 1.7; color: #4A5146; padding-left: 20px; margin: 0;">
                <li><strong>AI Content Studio:</strong> Constructs conversion-focused copywriting with customizable tone and structural hierarchy.</li>
                <li><strong>Customer Segmentation:</strong> Applies StandardScaler normalization and K-Means clustering across Recency, Frequency, and Monetary (RFM) metrics.</li>
                <li><strong>Engagement Prediction:</strong> Evaluates tabular features via an optimized Logistic Regression pipeline to forecast customer open and click probabilities.</li>
                <li><strong>Automated Campaign Dispatch:</strong> Enforces explicit user verification, subscriber consent validation, and duplicate send suppression.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_actions:
        st.markdown("""
        <div class="editorial-panel">
            <h3 class="editorial-panel-header">Workflow Guidance</h3>
            <p style="font-size: 13.5px; line-height: 1.6; color: #4A5146; margin: 0 0 12px 0;">
                Follow the standard lifecycle to execute an optimized marketing campaign:
            </p>
            <ol style="font-size: 13px; line-height: 1.6; color: #52604D; padding-left: 18px; margin: 0;">
                <li>Generate copy in <strong>AI Email Generator</strong>.</li>
                <li>Review cohort distributions in <strong>Segmentation</strong>.</li>
                <li>Verify propensity thresholds in <strong>Prediction</strong>.</li>
                <li>Approve and dispatch via <strong>Campaigns</strong>.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="page-intro" style="margin-bottom: 16px;">
        <h3 class="page-title" style="font-size: 22px;">Recent Campaign Records</h3>
    </div>
    """, unsafe_allow_html=True)
    
    recent_campaigns = get_all_campaigns()
    if recent_campaigns:
        camp_df = pd.DataFrame(recent_campaigns)
        camp_df = camp_df[[
            "campaign_id", "title", "target_segment", "status",
            "total_sent", "total_delivered", "total_opens", "total_clicks", "sent_at"
        ]]
        camp_df.columns = [
            "ID", "Campaign Title", "Segment", "Status",
            "Sent", "Delivered", "Opens", "Clicks", "Dispatched At"
        ]
        st.dataframe(camp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No campaigns executed yet. Navigate to Email Campaigns & Analytics to create your first dispatch.")


# =========================================================
# PAGE 2: AI EMAIL GENERATOR
# =========================================================
elif selected_page == "AI Email Generator":
    st.markdown("""
    <div class="page-intro">
        <div class="page-kicker">CONTENT STUDIO</div>
        <h2 class="page-title">AI Email Generator</h2>
        <p class="page-deck">Generate and customize conversion-focused marketing copy with tone adaptation and personalized attributes.</p>
    </div>
    """, unsafe_allow_html=True)

    gen_col1, gen_col2 = st.columns([1, 1], gap="large")

    with gen_col1:
        st.markdown("""
        <div class="editorial-panel">
            <h3 class="editorial-panel-header">Generation Parameters</h3>
        """, unsafe_allow_html=True)
        
        prompt_input = st.text_area(
            "Campaign Objective / Topic",
            value="Introduce our new Enterprise Cloud & AI Security Suite with priority onboarding and 20% discount for annual plans.",
            height=110,
            help="Describe the message, offer, key points, or target benefits."
        )

        tone_choice = st.selectbox("Writing Tone", AVAILABLE_TONES, index=0)
        interest_choice = st.selectbox("Target Audience Interest", CUSTOMER_INTERESTS, index=0)
        product_name = st.text_input("Product / Service Name", value="IntelliSuite Enterprise")
        cta_text = st.text_input("Call to Action (CTA) Button Text", value="Claim Your Priority Access")

        btn_generate = st.button("Generate Marketing Email", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if btn_generate:
            with st.spinner("Generating marketing copy..."):
                generator = AIEmailGenerator()
                generated = generator.generate_email(
                    prompt=prompt_input,
                    tone=tone_choice,
                    target_interest=interest_choice,
                    product_name=product_name,
                    call_to_action=cta_text
                )
                st.session_state.current_generated_email = generated
                st.success("Draft generated successfully.")

    with gen_col2:
        st.markdown("""
        <div class="editorial-panel">
            <h3 class="editorial-panel-header">Draft Review & Editor</h3>
        """, unsafe_allow_html=True)
        
        current_sub = st.session_state.current_generated_email.get("subject", "")
        editable_subject = st.text_input("Email Subject Line", value=current_sub)

        alts = st.session_state.current_generated_email.get("alternative_subjects", [])
        if alts:
            st.caption("Alternative Subject Suggestions:")
            for alt in alts:
                st.markdown(f"- `{alt}`")

        current_body = st.session_state.current_generated_email.get("body", "")
        editable_body = st.text_area("Email Body Copy", value=current_body, height=270)

        save_col, tpl_col = st.columns(2)
        with save_col:
            if st.button("Save as Template", use_container_width=True):
                save_template(
                    title=editable_subject[:30],
                    subject=editable_subject,
                    body=editable_body,
                    tone=tone_choice,
                    target_interest=interest_choice
                )
                st.success("Template saved successfully.")
        with tpl_col:
            if st.button("Update Active Draft", use_container_width=True):
                st.session_state.current_generated_email["subject"] = editable_subject
                st.session_state.current_generated_email["body"] = editable_body
                st.info("Active draft updated.")
        
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE 3: CUSTOMER SEGMENTATION
# =========================================================
elif selected_page == "Customer Segmentation":
    st.markdown("""
    <div class="page-intro">
        <div class="page-kicker">AUDIENCE INTELLIGENCE</div>
        <h2 class="page-title">Customer Segmentation</h2>
        <p class="page-deck">Partition customer bases into actionable cohorts based on standardized Recency, Frequency, and Monetary (RFM) dynamics.</p>
    </div>
    """, unsafe_allow_html=True)

    seg_controls_col1, seg_controls_col2 = st.columns([1, 1], gap="large")

    with seg_controls_col1:
        st.markdown("""
        <div class="editorial-panel">
            <h3 class="editorial-panel-header">Data Ingestion</h3>
        """, unsafe_allow_html=True)
        
        uploaded_csv = st.file_uploader("Upload Customer CSV (Optional)", type=["csv"])

        if uploaded_csv is not None:
            try:
                user_df = pd.read_csv(uploaded_csv)
                st.success(f"Uploaded file loaded with {len(user_df)} records.")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                user_df = None
        else:
            user_df = None

        if st.button("Generate Customer Dataset (3,000 Records)", use_container_width=True):
            with st.spinner("Generating 3,000 customer records with realistic distributions..."):
                syn_df = generate_synthetic_customers(n=3000, seed=int(datetime.now().timestamp()) % 1000)
                clustered_syn_df, _, _, _ = perform_rfm_clustering(syn_df, n_clusters=4, save_models=True)
                initial_probs = predict_customer_engagement(clustered_syn_df, pipeline=None)
                clustered_syn_df["engagement_score"] = initial_probs
                upsert_customers_from_df(clustered_syn_df, replace_existing=True)
                st.success("Dataset generated and stored successfully.")
                st.rerun()
                
        st.markdown("</div>", unsafe_allow_html=True)

    with seg_controls_col2:
        st.markdown("""
        <div class="editorial-panel">
            <h3 class="editorial-panel-header">Clustering Parameters</h3>
        """, unsafe_allow_html=True)
        
        k_clusters = st.slider("Number of Clusters (k)", min_value=3, max_value=6, value=4)
        
        if st.button("Run K-Means Clustering", type="primary", use_container_width=True):
            current_customers = get_all_customers_df()
            if not current_customers.empty:
                with st.spinner("Applying StandardScaler and K-Means clustering..."):
                    clustered_df, scaler, kmeans, summary = perform_rfm_clustering(
                        current_customers,
                        n_clusters=k_clusters,
                        save_models=True
                    )
                    update_customer_segments(clustered_df)
                    st.success(f"K-Means clustering completed with k={k_clusters}.")
                    st.rerun()
                    
        st.markdown("</div>", unsafe_allow_html=True)

    customers_df = get_all_customers_df()

    if not customers_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="page-intro" style="margin-bottom: 20px;">
            <h3 class="page-title" style="font-size: 22px;">Cohort Distribution & Monetary Valuation</h3>
        </div>
        """, unsafe_allow_html=True)

        c_chart1, c_chart2 = st.columns(2, gap="medium")
        with c_chart1:
            st.plotly_chart(plot_segment_distribution(customers_df), use_container_width=True)
        with c_chart2:
            st.plotly_chart(plot_segment_monetary_comparison(customers_df), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="page-intro" style="margin-bottom: 20px;">
            <h3 class="page-title" style="font-size: 22px;">3D Interactive RFM Feature Space</h3>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(plot_rfm_scatter_3d(customers_df), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="page-intro" style="margin-bottom: 16px;">
            <h3 class="page-title" style="font-size: 22px;">Segment Statistical Profiles</h3>
        </div>
        """, unsafe_allow_html=True)
        summary_profile = customers_df.groupby("segment_name").agg(
            Count=("customer_id", "count"),
            Avg_Recency_Days=("recency", lambda x: round(x.mean(), 1)),
            Avg_Frequency=("frequency", lambda x: round(x.mean(), 1)),
            Avg_Spend_USD=("monetary", lambda x: f"${round(x.mean(), 2):,.2f}"),
            Total_Spend_USD=("monetary", lambda x: f"${round(x.sum(), 2):,.2f}"),
            Subscribed=("is_subscribed", "sum")
        ).reset_index()
        st.dataframe(summary_profile, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="page-intro" style="margin-bottom: 16px;">
            <h3 class="page-title" style="font-size: 22px;">Audience Directory</h3>
        </div>
        """, unsafe_allow_html=True)
        filter_seg = st.selectbox("Filter Directory by Segment", ["All"] + list(customers_df["segment_name"].unique()))
        
        display_df = customers_df if filter_seg == "All" else customers_df[customers_df["segment_name"] == filter_seg]
        st.dataframe(
            display_df[[
                "customer_id", "name", "email", "interest", "company",
                "recency", "frequency", "monetary", "segment_name", "is_subscribed"
            ]].head(200),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PAGE 4: ENGAGEMENT PREDICTION
# =========================================================
elif selected_page == "Engagement Prediction":
    st.markdown("""
    <div class="page-intro">
        <div class="page-kicker">PROPENSITY MODELING</div>
        <h2 class="page-title">Customer Engagement Prediction</h2>
        <p class="page-deck">Estimate the likelihood of recipients opening or clicking upcoming campaigns using an optimized Logistic Regression pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    pred_col1, pred_col2 = st.columns([1, 1], gap="large")

    with pred_col1:
        st.markdown("""
        <div class="editorial-panel">
            <h3 class="editorial-panel-header">Model Training Controls</h3>
        """, unsafe_allow_html=True)
        
        test_size = st.slider("Test Split Proportion", min_value=0.10, max_value=0.35, value=0.20, step=0.05)
        
        btn_train = st.button("Train Logistic Regression Model", type="primary", use_container_width=True)
        if btn_train:
            c_df = get_all_customers_df()
            if not c_df.empty:
                with st.spinner("Fitting StandardScaler and Logistic Regression pipeline..."):
                    results = train_engagement_model(c_df, test_size=test_size, save_model=True)
                    st.session_state.last_training_results = results

                    scores_dict = dict(zip(c_df["customer_id"], results["full_probabilities"]))
                    update_customer_engagement_scores(scores_dict)

                    st.success("Model trained and serialized successfully.")
                    
        st.markdown("</div>", unsafe_allow_html=True)

    with pred_col2:
        st.markdown("""
        <div class="editorial-panel">
            <h3 class="editorial-panel-header">Pipeline Specification</h3>
            <ul style="font-size: 13.5px; line-height: 1.7; color: #4A5146; padding-left: 20px; margin: 0;">
                <li><strong>Algorithm:</strong> Scikit-learn Logistic Regression (L-BFGS solver, L2 penalty).</li>
                <li><strong>Pre-processing:</strong> StandardScaler zero-mean unit-variance scaling.</li>
                <li><strong>Feature Set:</strong> Recency, Frequency, Monetary, Previous Open Rate, Previous Click Rate.</li>
                <li><strong>Inference Output:</strong> Engagement probability spectrum (0.00 &ndash; 1.00).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    train_res = st.session_state.last_training_results
    if train_res:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="page-intro" style="margin-bottom: 20px;">
            <h3 class="page-title" style="font-size: 22px;">Validation & Classification Metrics</h3>
        </div>
        """, unsafe_allow_html=True)

        met = train_res["metrics"]
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-kicker">Accuracy</div>
                <div class="metric-val">{met['accuracy']*100:.1f}%</div>
                <div class="metric-caption">Test Cohort</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-kicker">Precision</div>
                <div class="metric-val">{met['precision']*100:.1f}%</div>
                <div class="metric-caption">Positive Predictive</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-kicker">Recall</div>
                <div class="metric-val">{met['recall']*100:.1f}%</div>
                <div class="metric-caption">Sensitivity</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-kicker">F1-Score</div>
                <div class="metric-val">{met['f1_score']:.3f}</div>
                <div class="metric-caption">Harmonic Mean</div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-kicker">ROC-AUC</div>
                <div class="metric-val">{met['roc_auc']:.3f}</div>
                <div class="metric-caption">Discrimination</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ch_col1, ch_col2 = st.columns(2, gap="medium")
        with ch_col1:
            st.plotly_chart(plot_confusion_matrix(train_res["confusion_matrix"]), use_container_width=True)
        with ch_col2:
            st.plotly_chart(plot_roc_curve(
                train_res["roc_curve"]["fpr"],
                train_res["roc_curve"]["tpr"],
                train_res["roc_curve"]["auc"]
            ), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="page-intro" style="margin-bottom: 20px;">
            <h3 class="page-title" style="font-size: 22px;">Standardized Feature Weights & Odds Ratios</h3>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(plot_feature_importance(train_res["feature_importance"]), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="page-intro" style="margin-bottom: 16px;">
        <h3 class="page-title" style="font-size: 22px;">Scored Audience Directory</h3>
    </div>
    """, unsafe_allow_html=True)
    
    all_cust = get_all_customers_df()
    if not all_cust.empty:
        min_prob_filter = st.slider("Filter by Minimum Engagement Probability Threshold:", 0.0, 1.0, 0.50, step=0.05)
        filtered_scored = all_cust[all_cust["engagement_score"] >= min_prob_filter]
        
        st.caption(f"Displaying {len(filtered_scored):,} of {len(all_cust):,} records meeting probability threshold (&ge; {min_prob_filter:.2f}).")
        
        st.dataframe(
            filtered_scored[[
                "customer_id", "name", "email", "interest", "segment_name",
                "engagement_score", "recency", "frequency", "previous_open_rate", "is_subscribed"
            ]].head(150),
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PAGE 5: EMAIL CAMPAIGNS & ANALYTICS
# =========================================================
elif selected_page == "Email Campaigns & Analytics":
    st.markdown("""
    <div class="page-intro">
        <div class="page-kicker">DISPATCH & TELEMETRY</div>
        <h2 class="page-title">Email Campaigns & Analytics</h2>
        <p class="page-deck">Target audiences with consent safeguards, dispatch approved campaigns, and review verified delivery telemetry.</p>
    </div>
    """, unsafe_allow_html=True)

    camp_create_tab, camp_analytics_tab, camp_consent_tab = st.tabs([
        "Create & Dispatch Campaign",
        "Campaign Analytics & Logs",
        "Consent & Unsubscribe Manager"
    ])

    # SUB-TAB 1: CREATE & DISPATCH
    with camp_create_tab:
        st.markdown("""
        <div class="editorial-panel">
            <h3 class="editorial-panel-header">1. Target Audience Calibration</h3>
        """, unsafe_allow_html=True)
        
        all_customers = get_all_customers_df()

        target_col1, target_col2, target_col3 = st.columns(3)
        with target_col1:
            segment_options = ["All"] + (list(all_customers["segment_name"].unique()) if not all_customers.empty else [])
            selected_segment = st.selectbox("Target Segment", segment_options, index=0)
        with target_col2:
            min_score = st.slider("Min Engagement Propensity", 0.0, 1.0, 0.40, step=0.05)
        with target_col3:
            max_recipients = st.number_input("Recipient Batch Cap", min_value=1, max_value=5000, value=50)

        recipients = all_customers.copy()
        if selected_segment != "All":
            recipients = recipients[recipients["segment_name"] == selected_segment]
        recipients = recipients[recipients["engagement_score"] >= min_score]

        unsub_in_pool = len(recipients[recipients["is_subscribed"] == 0])
        eligible_recipients = recipients[recipients["is_subscribed"] == 1].head(max_recipients)

        st.markdown(f"""
        <div style="background-color: #F4EFEA; border: 1px solid #E8E3D9; border-left: 3px solid #4E5F47; padding: 12px 16px; border-radius: 4px; margin-top: 10px; font-size: 13.5px; color: #3B4238;">
            <strong>Audience Specification:</strong> <strong>{len(eligible_recipients)}</strong> eligible consented recipients targeted. 
            (<em>{unsub_in_pool} unsubscribed contact(s) automatically excluded</em>).
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="editorial-panel">
            <h3 class="editorial-panel-header">2. Campaign Content & Message</h3>
        """, unsafe_allow_html=True)
        
        templates = get_templates()
        if templates:
            tpl_choice = st.selectbox(
                "Load Saved Template:",
                options=["(Use Active Draft)"] + [f"{t['title']} ({t['tone']})" for t in templates]
            )
            if tpl_choice != "(Use Active Draft)":
                sel_tpl = next(t for t in templates if f"{t['title']} ({t['tone']})" == tpl_choice)
                camp_subject = st.text_input("Campaign Subject Line", value=sel_tpl["subject"])
                camp_body = st.text_area("Campaign Body Copy", value=sel_tpl["body"], height=160)
            else:
                camp_subject = st.text_input("Campaign Subject Line", value=st.session_state.current_generated_email.get("subject", ""))
                camp_body = st.text_area("Campaign Body Copy", value=st.session_state.current_generated_email.get("body", ""), height=160)
        else:
            camp_subject = st.text_input("Campaign Subject Line", value=st.session_state.current_generated_email.get("subject", ""))
            camp_body = st.text_area("Campaign Body Copy", value=st.session_state.current_generated_email.get("body", ""), height=160)

        camp_title = st.text_input("Internal Campaign Identifier", value=f"Campaign_{selected_segment.replace(' ', '_')}_{datetime.now().strftime('%b%d')}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="editorial-panel">
            <h3 class="editorial-panel-header">3. Dispatch Timing & Verification</h3>
        """, unsafe_allow_html=True)
        
        schedule_option = st.radio("Dispatch Scheduling", ["Immediate Send", "Schedule for Later"], horizontal=True)
        scheduled_time = None
        if schedule_option == "Schedule for Later":
            scheduled_date = st.date_input("Scheduled Date", datetime.now() + timedelta(days=1))
            scheduled_time_val = st.time_input("Scheduled Time", datetime.now().time())
            scheduled_time = f"{scheduled_date} {scheduled_time_val.strftime('%H:%M:%S')}"

        approval_confirmed = st.checkbox(
            "I explicitly approve the dispatch of this marketing campaign to verified recipients and confirm compliance with recipient consent policies.",
            value=False
        )

        btn_dispatch = st.button("Dispatch Marketing Campaign", type="primary", disabled=not approval_confirmed, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if btn_dispatch:
            if not approval_confirmed:
                st.error("Approval required before sending.")
            elif len(eligible_recipients) == 0:
                st.error("No eligible recipients match the selected criteria.")
            else:
                with st.spinner("Processing campaign dispatch with duplicate check and compliance logging..."):
                    service = EmailAutomationService()
                    campaign_id = f"CAMP-{uuid.uuid4().hex[:8].upper()}"
                    
                    result = service.dispatch_campaign(
                        campaign_id=campaign_id,
                        title=camp_title,
                        subject_template=camp_subject,
                        body_template=camp_body,
                        recipients_df=eligible_recipients,
                        target_segment=selected_segment,
                        target_min_prob=min_score,
                        is_approved=approval_confirmed,
                        scheduled_at=scheduled_time
                    )

                    if result.get("status") == "Scheduled":
                        st.success(f"Campaign scheduled for {scheduled_time}.")
                    else:
                        st.success(f"Campaign dispatched successfully. Sent: {result['sent_count']}, Delivered: {result['delivered_count']}, Opens: {result['opened_count']} ({result['open_rate']}%), Clicks: {result['clicked_count']} ({result['click_rate']}%)")
                        if result["skipped_duplicate"] > 0:
                            st.warning(f"Prevented {result['skipped_duplicate']} duplicate email(s) from re-sending.")

    # SUB-TAB 2: CAMPAIGN ANALYTICS & LOGS
    with camp_analytics_tab:
        st.markdown("""
        <div class="page-intro" style="margin-bottom: 20px;">
            <h3 class="page-title" style="font-size: 22px;">Campaign Performance Analytics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        all_camps = get_all_campaigns()

        if all_camps:
            st.plotly_chart(plot_campaign_comparison(all_camps), use_container_width=True)

            selected_camp_id = st.selectbox(
                "Filter Detailed Audit Log by Campaign:",
                options=["All Campaigns"] + [c["campaign_id"] for c in all_camps],
                format_func=lambda cid: cid if cid == "All Campaigns" else f"{cid} — {next(c['title'] for c in all_camps if c['campaign_id'] == cid)}"
            )

            filter_cid = None if selected_camp_id == "All Campaigns" else selected_camp_id
            logs_df = get_campaign_logs(campaign_id=filter_cid)

            if not logs_df.empty:
                st.markdown(f"**Audit Logs ({len(logs_df)} records):**")
                st.dataframe(
                    logs_df[[
                        "campaign_id", "customer_id", "recipient_email", "subject",
                        "delivery_status", "opened", "clicked", "sent_at"
                    ]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No delivery logs found for the selected campaign.")
        else:
            st.info("No campaigns executed yet.")

    # SUB-TAB 3: CONSENT & UNSUBSCRIBE MANAGER
    with camp_consent_tab:
        st.markdown("""
        <div class="page-intro" style="margin-bottom: 20px;">
            <h3 class="page-title" style="font-size: 22px;">Recipient Consent & Unsubscribe Registry</h3>
            <p class="page-deck">Rigorous compliance protocols strictly suppress unsubscribed contacts and guarantee idempotent duplicate prevention.</p>
        </div>
        """, unsafe_allow_html=True)

        manage_col1, manage_col2 = st.columns(2, gap="large")

        with manage_col1:
            st.markdown("""
            <div class="editorial-panel">
                <h3 class="editorial-panel-header">Record Unsubscribe Request</h3>
            """, unsafe_allow_html=True)
            test_email_unsub = st.text_input("Enter customer email to unsubscribe:", placeholder="customer@example.com")
            if st.button("Record Unsubscribe", use_container_width=True):
                if test_email_unsub:
                    if unsubscribe_customer(test_email_unsub):
                        st.success(f"{test_email_unsub} has been unsubscribed. Subsequent campaigns will skip this contact.")
                    else:
                        st.error("Customer email not found.")
            st.markdown("</div>", unsafe_allow_html=True)

        with manage_col2:
            st.markdown("""
            <div class="editorial-panel">
                <h3 class="editorial-panel-header">Restore Subscription Consent</h3>
            """, unsafe_allow_html=True)
            test_email_resub = st.text_input("Enter customer email to resubscribe:", placeholder="customer@example.com")
            if st.button("Restore Subscription", use_container_width=True):
                if test_email_resub:
                    if resubscribe_customer(test_email_resub):
                        st.success(f"{test_email_resub} has been re-subscribed with consent.")
                    else:
                        st.error("Customer email not found.")
            st.markdown("</div>", unsafe_allow_html=True)

        c_all = get_all_customers_df()
        if not c_all.empty:
            unsub_list = c_all[c_all["is_subscribed"] == 0]
            st.markdown(f"**Suppression Registry ({len(unsub_list)} records protected from sends):**")
            st.dataframe(
                unsub_list[["customer_id", "name", "email", "interest", "company", "segment_name"]],
                use_container_width=True,
                hide_index=True
            )
