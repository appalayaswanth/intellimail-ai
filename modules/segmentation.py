"""Module 2: Customer Segmentation
Handles CSV upload, synthetic 3,000-customer dataset generation,
RFM analysis, StandardScaler normalization, and K-Means clustering with Plotly visualizations.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, Any

from config import (
    SCALER_MODEL_PATH,
    KMEANS_MODEL_PATH,
    CUSTOMER_INTERESTS,
    BRAND_COLORS
)


def generate_synthetic_customers(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic dataset of ~3,000 customers with RFM and engagement features.
    
    Identified clearly as synthetic data for academic and evaluation purposes.
    """
    np.random.seed(seed)

    first_names = [
        "Aarav", "Aditi", "Rahul", "Priya", "Ananya", "Rohan", "Sneha", "Vikram",
        "Pooja", "Arjun", "Kavya", "Siddharth", "Meera", "Kabir", "Neha", "Amit",
        "Alex", "Sophia", "Daniel", "Emma", "Liam", "Olivia", "Noah", "Ava",
        "Ethan", "Mia", "James", "Isabella", "Lucas", "Charlotte"
    ]
    last_names = [
        "Sharma", "Verma", "Patel", "Gupta", "Reddy", "Mehta", "Iyer", "Nair",
        "Singh", "Chopra", "Deshmukh", "Kulkarni", "Bose", "Joshi", "Das",
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson"
    ]
    company_suffixes = [
        "Tech", "Labs", "Solutions", "Ventures", "Systems", "Global",
        "Consulting", "Innovations", "Cloud", "Analytics", "Networks"
    ]

    customer_ids = [f"CUST-{i:05d}" for i in range(1, n + 1)]
    names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n)]

    # Clean domain emails
    emails = []
    seen_emails = set()
    for name in names:
        clean_name = "".join(ch for ch in name.lower() if ch.isalnum())
        base_email = f"{clean_name}@example.com"
        suffix = 1
        email = base_email
        while email in seen_emails:
            email = f"{clean_name}{suffix}@example.com"
            suffix += 1
        seen_emails.add(email)
        emails.append(email)

    companies = [f"{np.random.choice(last_names)} {np.random.choice(company_suffixes)}" for _ in range(n)]
    interests = np.random.choice(CUSTOMER_INTERESTS, size=n, p=[0.18, 0.16, 0.14, 0.15, 0.12, 0.10, 0.08, 0.07])

    # Realistic multivariate RFM distributions (mimicking true customer cohorts)
    # Cohort 1: Champions (15%) - Low recency, High freq, High monetary
    # Cohort 2: Loyal (30%) - Moderate recency, Moderate-high freq, Good monetary
    # Cohort 3: Promising (30%) - Low-to-mid recency, Low freq, Modest monetary
    # Cohort 4: At-Risk (25%) - High recency, Varied freq, Declining monetary

    cohorts = np.random.choice([0, 1, 2, 3], size=n, p=[0.15, 0.30, 0.30, 0.25])

    recency = np.zeros(n, dtype=int)
    frequency = np.zeros(n, dtype=int)
    monetary = np.zeros(n, dtype=float)
    prev_open_rate = np.zeros(n, dtype=float)
    prev_click_rate = np.zeros(n, dtype=float)

    for i in range(n):
        c = cohorts[i]
        if c == 0:  # Champions
            recency[i] = int(np.clip(np.random.exponential(scale=18) + 2, 1, 60))
            frequency[i] = int(np.clip(np.random.poisson(lam=28) + 10, 10, 60))
            monetary[i] = float(np.round(np.random.normal(loc=2800, scale=800), 2))
            prev_open_rate[i] = float(np.clip(np.random.beta(a=8, b=3), 0.30, 0.95))
            prev_click_rate[i] = float(np.clip(prev_open_rate[i] * np.random.uniform(0.35, 0.70), 0.05, 0.60))
        elif c == 1:  # Loyal
            recency[i] = int(np.clip(np.random.exponential(scale=45) + 15, 10, 120))
            frequency[i] = int(np.clip(np.random.poisson(lam=14) + 4, 4, 30))
            monetary[i] = float(np.round(np.random.normal(loc=1200, scale=400), 2))
            prev_open_rate[i] = float(np.clip(np.random.beta(a=5, b=4), 0.15, 0.75))
            prev_click_rate[i] = float(np.clip(prev_open_rate[i] * np.random.uniform(0.20, 0.50), 0.02, 0.40))
        elif c == 2:  # Promising
            recency[i] = int(np.clip(np.random.exponential(scale=35) + 5, 2, 90))
            frequency[i] = int(np.clip(np.random.poisson(lam=4) + 1, 1, 8))
            monetary[i] = float(np.round(np.random.normal(loc=380, scale=150), 2))
            prev_open_rate[i] = float(np.clip(np.random.beta(a=3, b=5), 0.05, 0.60))
            prev_click_rate[i] = float(np.clip(prev_open_rate[i] * np.random.uniform(0.10, 0.40), 0.00, 0.25))
        else:  # At-Risk
            recency[i] = int(np.clip(np.random.uniform(120, 365), 90, 365))
            frequency[i] = int(np.clip(np.random.poisson(lam=3) + 1, 1, 15))
            monetary[i] = float(np.round(np.random.normal(loc=420, scale=250), 2))
            prev_open_rate[i] = float(np.clip(np.random.beta(a=2, b=8), 0.01, 0.35))
            prev_click_rate[i] = float(np.clip(prev_open_rate[i] * np.random.uniform(0.00, 0.25), 0.00, 0.10))

    # Guard negative values
    monetary = np.clip(monetary, 25.0, 9999.0)

    # 95% subscribed, 5% unsubscribed for compliance demonstrations
    is_subscribed = np.random.choice([1, 0], size=n, p=[0.95, 0.05])

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "name": names,
        "email": emails,
        "interest": interests,
        "company": companies,
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "previous_open_rate": np.round(prev_open_rate, 4),
        "previous_click_rate": np.round(prev_click_rate, 4),
        "segment_id": -1,
        "segment_name": "Unsegmented",
        "engagement_score": 0.0,
        "is_subscribed": is_subscribed
    })

    return df


def perform_rfm_clustering(
    df: pd.DataFrame,
    n_clusters: int = 4,
    save_models: bool = True
) -> Tuple[pd.DataFrame, StandardScaler, KMeans, pd.DataFrame]:
    """Execute RFM normalization with StandardScaler and K-Means clustering.
    
    Dynamically maps clusters to human-interpretable segment names:
    - Champions (High Value)
    - Loyal Customers
    - Promising Prospects
    - At-Risk / Inactive
    """
    rfm_cols = ["recency", "frequency", "monetary"]

    # Validate columns
    for col in rfm_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from customer data.")

    # Apply StandardScaler
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df[rfm_cols])

    # Apply K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42
    )
    raw_clusters = kmeans.fit_predict(rfm_scaled)

    df_clustered = df.copy()
    df_clustered["raw_cluster"] = raw_clusters

    # Calculate cluster centroids in unscaled space to interpret and assign labels
    cluster_means = df_clustered.groupby("raw_cluster")[rfm_cols].mean()

    # Composite RFM Score for ranking: Higher Frequency and Monetary = Good, Lower Recency = Good
    # Score = (Mean Frequency z-score) + (Mean Monetary z-score) - (Mean Recency z-score)
    cluster_scores = {}
    for c in range(n_clusters):
        r_val = cluster_means.loc[c, "recency"]
        f_val = cluster_means.loc[c, "frequency"]
        m_val = cluster_means.loc[c, "monetary"]
        # Standardized ranking metric
        score = (f_val / (cluster_means["frequency"].mean() + 1e-5)) + \
                (m_val / (cluster_means["monetary"].mean() + 1e-5)) - \
                (r_val / (cluster_means["recency"].mean() + 1e-5))
        cluster_scores[c] = score

    # Sort cluster IDs by score descending
    sorted_clusters = sorted(cluster_scores.keys(), key=lambda k: cluster_scores[k], reverse=True)

    label_presets = [
        "Champions (High Value)",
        "Loyal Customers",
        "Promising Prospects",
        "At-Risk / Inactive"
    ]

    cluster_to_segment = {}
    for rank, orig_id in enumerate(sorted_clusters):
        label = label_presets[rank] if rank < len(label_presets) else f"Segment {rank+1}"
        cluster_to_segment[orig_id] = (rank, label)

    df_clustered["segment_id"] = df_clustered["raw_cluster"].map(lambda x: cluster_to_segment[x][0])
    df_clustered["segment_name"] = df_clustered["raw_cluster"].map(lambda x: cluster_to_segment[x][1])
    df_clustered = df_clustered.drop(columns=["raw_cluster"])

    # Cluster summary
    summary_df = df_clustered.groupby("segment_name").agg(
        Count=("customer_id", "count"),
        Avg_Recency_Days=("recency", lambda x: round(x.mean(), 1)),
        Avg_Frequency=("frequency", lambda x: round(x.mean(), 1)),
        Avg_Monetary_USD=("monetary", lambda x: round(x.mean(), 2)),
        Total_Monetary_USD=("monetary", lambda x: round(x.sum(), 2)),
        Avg_Open_Rate=("previous_open_rate", lambda x: f"{round(x.mean()*100, 1)}%"),
        Subscribed_Count=("is_subscribed", "sum")
    ).reset_index()

    if save_models:
        joblib.dump(scaler, SCALER_MODEL_PATH)
        joblib.dump(kmeans, KMEANS_MODEL_PATH)

    return df_clustered, scaler, kmeans, summary_df


def plot_rfm_scatter_3d(df: pd.DataFrame) -> go.Figure:
    """Generate an interactive 3D scatter plot of Recency, Frequency, and Monetary."""
    sample_df = df.sample(min(len(df), 1200), random_state=42) if len(df) > 1200 else df

    color_map = {
        "Champions (High Value)": "#46634B",    # Muted Olive
        "Loyal Customers": "#324436",          # Deep Charcoal Olive
        "Promising Prospects": "#8F754E",      # Warm Ochre
        "At-Risk / Inactive": "#935353"        # Muted Terracotta
    }

    fig = px.scatter_3d(
        sample_df,
        x="recency",
        y="frequency",
        z="monetary",
        color="segment_name",
        color_discrete_map=color_map,
        hover_name="name",
        hover_data=["customer_id", "interest", "company"],
        labels={
            "recency": "Recency (Days)",
            "frequency": "Frequency (Orders)",
            "monetary": "Monetary ($)"
        },
        title="3D Customer RFM Space by Segment",
        opacity=0.75
    )

    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#1F241E"),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(title="Recency (Days)", backgroundcolor="#FAF8F5", gridcolor="#E8E4DC"),
            yaxis=dict(title="Frequency", backgroundcolor="#FAF8F5", gridcolor="#E8E4DC"),
            zaxis=dict(title="Monetary ($)", backgroundcolor="#FAF8F5", gridcolor="#E8E4DC"),
            bgcolor="#FAF8F5"
        ),
        legend=dict(title="Segment", orientation="h", y=-0.1),
        margin=dict(l=0, r=0, b=0, t=40),
        height=520
    )
    return fig


def plot_segment_distribution(df: pd.DataFrame) -> go.Figure:
    """Generate a donut chart showing customer distribution across segments."""
    segment_counts = df["segment_name"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]

    color_map = {
        "Champions (High Value)": "#46634B",
        "Loyal Customers": "#324436",
        "Promising Prospects": "#8F754E",
        "At-Risk / Inactive": "#935353"
    }

    fig = px.pie(
        segment_counts,
        names="Segment",
        values="Count",
        color="Segment",
        color_discrete_map=color_map,
        hole=0.55,
        title="Customer Segment Distribution"
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#FAF8F5", width=2))
    )
    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#1F241E"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=20, r=20, b=20, t=40),
        height=380
    )
    return fig


def plot_segment_monetary_comparison(df: pd.DataFrame) -> go.Figure:
    """Generate a bar chart comparing Average and Total Monetary Value per segment."""
    summary = df.groupby("segment_name")["monetary"].agg(["mean", "sum"]).reset_index()
    summary.columns = ["Segment", "Average_Spent", "Total_Spent"]

    fig = px.bar(
        summary,
        x="Segment",
        y="Average_Spent",
        color="Segment",
        text_auto="$.2f",
        title="Average Customer Spend by Segment ($)",
        color_discrete_sequence=["#46634B", "#324436", "#8F754E", "#935353"]
    )
    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#1F241E"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Average Spend ($)", gridcolor="#E8E4DC"),
        xaxis=dict(gridcolor="#E8E4DC"),
        showlegend=False,
        margin=dict(l=20, r=20, b=20, t=40),
        height=380
    )
    return fig
