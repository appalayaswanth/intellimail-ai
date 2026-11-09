"""Module 3: Customer Engagement Prediction
Trains a Scikit-learn Logistic Regression model on CPU to predict
whether a customer is likely to open/engage with an upcoming email campaign.
Evaluates accuracy, precision, recall, F1, confusion matrix, ROC-AUC, and feature odds ratios.
Saves model with Joblib. Clearly marks synthetic and illustrative nature for academic rigor.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
import joblib
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Tuple, Optional

from config import ENGAGEMENT_MODEL_PATH


FEATURE_COLS = [
    "recency",
    "frequency",
    "monetary",
    "previous_open_rate",
    "previous_click_rate"
]

FEATURE_LABELS = {
    "recency": "Recency (Days Ago)",
    "frequency": "Purchase Frequency",
    "monetary": "Monetary Spend ($)",
    "previous_open_rate": "Historical Open Rate",
    "previous_click_rate": "Historical Click Rate"
}


def synthesize_engagement_labels(df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Generate statistically grounded synthetic engagement ground truth labels (0 = No, 1 = Engaged).
    
    Uses realistic logistic link with domain coefficients + noise:
    - High frequency & high historical rates boost engagement
    - High recency (inactivity) penalizes engagement
    - Identified clearly as synthetic data for academic project demonstrations.
    """
    np.random.seed(seed)
    r = df["recency"].values
    f = df["frequency"].values
    m = df["monetary"].values
    o = df["previous_open_rate"].values
    c = df["previous_click_rate"].values

    # Standardized features to build realistic logit
    logit = (
        -0.8
        - 0.012 * r               # Inactive customers less likely to open
        + 0.065 * f               # Frequent buyers more likely to open
        + 0.0003 * m              # High spenders slightly more engaged
        + 3.8 * o                 # Historical open rate is strongest positive signal
        + 4.5 * c                 # Historical click rate is very strong signal
        + np.random.normal(0, 0.4, size=len(df))  # Natural behavioral variance
    )

    prob = 1.0 / (1.0 + np.exp(-logit))
    # Binary engagement outcome
    y = (prob >= 0.50).astype(int)
    return y


def train_engagement_model(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    save_model: bool = True
) -> Dict[str, Any]:
    """Train Scikit-learn Logistic Regression pipeline (StandardScaler + LogisticRegression) on CPU.
    
    Returns evaluation metrics, confusion matrix, ROC data, and pipeline.
    """
    # Ensure features exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: '{col}'")

    X = df[FEATURE_COLS].copy()
    y = synthesize_engagement_labels(df, seed=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scikit-learn Pipeline on CPU
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC Curve coordinates
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # Feature Importance (Standardized Coefficients & Odds Ratios)
    coefs = pipeline.named_steps["classifier"].coef_[0]
    odds_ratios = np.exp(coefs)
    feature_importance = pd.DataFrame({
        "Feature": [FEATURE_LABELS[f] for f in FEATURE_COLS],
        "Coefficient": coefs,
        "Odds_Ratio": odds_ratios
    }).sort_values(by="Coefficient", ascending=False)

    # Batch score entire input dataframe
    full_probs = pipeline.predict_proba(X)[:, 1]

    if save_model:
        joblib.dump(pipeline, ENGAGEMENT_MODEL_PATH)

    return {
        "pipeline": pipeline,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        },
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(auc, 4)},
        "feature_importance": feature_importance,
        "full_probabilities": full_probs
    }


def load_saved_model() -> Optional[Pipeline]:
    """Load previously persisted model from disk."""
    if ENGAGEMENT_MODEL_PATH.exists():
        try:
            return joblib.load(ENGAGEMENT_MODEL_PATH)
        except Exception:
            return None
    return None


def predict_customer_engagement(df: pd.DataFrame, pipeline: Optional[Pipeline] = None) -> np.ndarray:
    """Predict engagement probabilities for customers."""
    if pipeline is None:
        pipeline = load_saved_model()

    if pipeline is None:
        # Fallback heuristic calculation if model not trained yet
        r = df["recency"].values
        f = df["frequency"].values
        o = df["previous_open_rate"].values
        c = df["previous_click_rate"].values
        raw = 0.2 + (0.4 * o) + (0.3 * c) + (0.1 * np.clip(f / 20, 0, 1)) - (0.1 * np.clip(r / 180, 0, 1))
        return np.clip(np.round(raw, 4), 0.05, 0.98)

    X = df[FEATURE_COLS].copy()
    probs = pipeline.predict_proba(X)[:, 1]
    return np.round(probs, 4)


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """Plot interactive confusion matrix heatmap with annotations."""
    z = cm
    x = ["Predicted: Unengaged (0)", "Predicted: Engaged (1)"]
    y = ["Actual: Unengaged (0)", "Actual: Engaged (1)"]

    annotations = []
    for i, row in enumerate(z):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=f"<b>{value}</b>",
                    showarrow=False,
                    font=dict(color="#FAF8F5" if value > z.max() / 2 else "#1F241E", size=17, family="Plus Jakarta Sans, sans-serif")
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0, "#FAF8F5"], [0.4, "#C8D4C4"], [1, "#445340"]],
            showscale=False
        )
    )
    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#1F241E"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Confusion Matrix (Test Set)",
        annotations=annotations,
        margin=dict(l=40, r=40, b=40, t=50),
        height=360
    )
    return fig


def plot_roc_curve(fpr: list, tpr: list, auc: float) -> go.Figure:
    """Plot ROC-AUC curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"Logistic Regression (AUC = {auc:.3f})",
        line=dict(color="#4A5845", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Baseline (AUC = 0.500)",
        line=dict(color="#C2BCB2", dash="dash", width=1.5)
    ))

    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#1F241E"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Receiver Operating Characteristic (ROC)",
        xaxis=dict(title="False Positive Rate (1 - Specificity)", gridcolor="#E8E4DC"),
        yaxis=dict(title="True Positive Rate (Recall)", gridcolor="#E8E4DC"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=40, b=40, t=50),
        height=360
    )
    return fig


def plot_feature_importance(fi_df: pd.DataFrame) -> go.Figure:
    """Plot standardized logistic regression coefficients."""
    colors = ["#46634B" if c >= 0 else "#935353" for c in fi_df["Coefficient"]]

    fig = go.Figure(go.Bar(
        x=fi_df["Coefficient"],
        y=fi_df["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{c:+.2f} (OR: {o:.2f}x)" for c, o in zip(fi_df["Coefficient"], fi_df["Odds_Ratio"])],
        textposition="outside"
    ))

    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#1F241E"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Feature Impact on Engagement (Coefficients & Odds Ratio)",
        xaxis=dict(title="Standardized Impact", gridcolor="#E8E4DC"),
        yaxis=dict(gridcolor="#E8E4DC"),
        margin=dict(l=40, r=80, b=40, t=50),
        height=340
    )
    return fig
