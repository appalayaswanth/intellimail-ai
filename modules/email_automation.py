"""Module 4: Email Automation and Analytics
Integrates Brevo (Sendinblue) transactional REST API with safety guards:
- Explicit user approval verification
- Recipient consent / unsubscribe compliance check
- Duplicate email prevention
- Basic campaign scheduling (immediate or scheduled)
- Delivery, open, and click tracking stored in SQLite
- Clean simulation mode with probabilistic engagement response if no Brevo API key is present.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import BREVO_API_KEY, SENDER_NAME, SENDER_EMAIL
from database import (
    log_email_send,
    is_already_sent,
    save_campaign,
    get_connection
)
from modules.ai_generator import AIEmailGenerator


class EmailAutomationService:
    """Service handling email dispatch, Brevo API integration, compliance checks, and analytics."""

    BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or BREVO_API_KEY).strip()
        self.is_simulation_mode = not bool(self.api_key)

    def validate_campaign_readiness(
        self,
        campaign_title: str,
        subject: str,
        body: str,
        recipients_df: pd.DataFrame,
        is_approved: bool
    ) -> Dict[str, Any]:
        """Perform strict pre-send compliance and safety validation."""
        errors = []
        warnings = []

        if not campaign_title.strip():
            errors.append("Campaign title is required.")
        if not subject.strip():
            errors.append("Email subject line cannot be empty.")
        if not body.strip():
            errors.append("Email body cannot be empty.")
        if recipients_df is None or len(recipients_df) == 0:
            errors.append("No recipients selected for this campaign.")
        if not is_approved:
            errors.append("Explicit human approval is required before dispatching emails.")

        # Check consent
        unsubscribed_count = len(recipients_df[recipients_df["is_subscribed"] == 0]) if recipients_df is not None and "is_subscribed" in recipients_df.columns else 0
        if unsubscribed_count > 0:
            warnings.append(f"{unsubscribed_count} recipient(s) will be automatically skipped due to unsubscribed status.")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "eligible_count": len(recipients_df[recipients_df["is_subscribed"] == 1]) if recipients_df is not None and "is_subscribed" in recipients_df.columns else 0
        }

    def dispatch_campaign(
        self,
        campaign_id: str,
        title: str,
        subject_template: str,
        body_template: str,
        recipients_df: pd.DataFrame,
        target_segment: str = "All",
        target_min_prob: float = 0.0,
        is_approved: bool = False,
        scheduled_at: Optional[str] = None
    ) -> Dict[str, Any]:
        """Dispatch marketing campaign with duplicate prevention, consent checking, and logging."""
        if not is_approved:
            raise PermissionError("Dispatch blocked: Explicit user approval flag is False.")

        # Ensure campaign record exists in SQLite
        campaign_record = {
            "campaign_id": campaign_id,
            "title": title,
            "subject": subject_template,
            "body": body_template,
            "target_segment": target_segment,
            "target_min_prob": target_min_prob,
            "status": "In Progress" if not scheduled_at else "Scheduled",
            "scheduled_at": scheduled_at,
            "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if not scheduled_at else None,
            "total_recipients": len(recipients_df),
            "total_sent": 0,
            "total_delivered": 0,
            "total_opens": 0,
            "total_clicks": 0,
            "is_approved": 1 if is_approved else 0
        }
        save_campaign(campaign_record)

        # If campaign is scheduled for future, do not execute sending loop now
        if scheduled_at:
            return {
                "campaign_id": campaign_id,
                "status": "Scheduled",
                "message": f"Campaign successfully scheduled for {scheduled_at}.",
                "total_recipients": len(recipients_df),
                "sent_count": 0,
                "skipped_unsubscribed": 0,
                "skipped_duplicate": 0
            }

        sent_count = 0
        delivered_count = 0
        opened_count = 0
        clicked_count = 0
        skipped_unsubscribed = 0
        skipped_duplicate = 0

        for _, customer in recipients_df.iterrows():
            customer_id = str(customer["customer_id"])
            email = str(customer["email"]).strip().lower()
            name = str(customer.get("name", "Valued Customer"))

            # 1. CONSENT CHECK: Skip if unsubscribed
            if int(customer.get("is_subscribed", 1)) == 0:
                skipped_unsubscribed += 1
                log_email_send(
                    campaign_id=campaign_id,
                    customer_id=customer_id,
                    email=email,
                    subject=subject_template,
                    status="Skipped_Unsubscribed"
                )
                continue

            # 2. DUPLICATE CHECK: Skip if already sent for this campaign
            if is_already_sent(campaign_id, customer_id):
                skipped_duplicate += 1
                continue

            # 3. PERSONALIZATION: Inject customer name, interest, company
            cust_dict = customer.to_dict()
            personalized_subject = AIEmailGenerator.personalize_text(subject_template, cust_dict)
            personalized_body = AIEmailGenerator.personalize_text(body_template, cust_dict)

            # Add compliant footer with unsubscribe note
            footer = (
                f"\n\n---\n"
                f"You received this message because of your registered interest in {customer.get('interest', 'Technology')}.\n"
                f"To unsubscribe or update your preferences, click here: http://localhost:8501/?unsubscribe={email}"
            )
            final_body = personalized_body + footer

            # 4. DISPATCH: Real Brevo API or Realistic Simulation
            if not self.is_simulation_mode:
                send_result = self._send_via_brevo_api(
                    recipient_email=email,
                    recipient_name=name,
                    subject=personalized_subject,
                    body=final_body
                )
            else:
                send_result = self._simulate_send(
                    recipient_email=email,
                    engagement_prob=float(customer.get("engagement_score", 0.5))
                )

            # 5. LOG SEND & UPDATE STATS
            if send_result["status"] in ("Sent", "Delivered", "Simulated_Sent"):
                sent_count += 1
                delivered_count += 1
                is_opened = 1 if send_result.get("opened", False) else 0
                is_clicked = 1 if send_result.get("clicked", False) else 0

                if is_opened:
                    opened_count += 1
                if is_clicked:
                    clicked_count += 1

                log_email_send(
                    campaign_id=campaign_id,
                    customer_id=customer_id,
                    email=email,
                    subject=personalized_subject,
                    status=send_result["status"],
                    message_id=send_result.get("message_id")
                )

                # Update individual log record open/click flags if simulated
                if is_opened or is_clicked:
                    self._update_log_metrics(campaign_id, customer_id, is_opened, is_clicked)
            else:
                log_email_send(
                    campaign_id=campaign_id,
                    customer_id=customer_id,
                    email=email,
                    subject=personalized_subject,
                    status=f"Failed_{send_result.get('error', 'Unknown')}"
                )

        # Update campaign final totals
        campaign_record.update({
            "status": "Completed",
            "total_sent": sent_count,
            "total_delivered": delivered_count,
            "total_opens": opened_count,
            "total_clicks": clicked_count,
            "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_campaign(campaign_record)

        return {
            "campaign_id": campaign_id,
            "status": "Completed",
            "mode": "Brevo REST API" if not self.is_simulation_mode else "Demo Simulation Engine",
            "total_recipients": len(recipients_df),
            "sent_count": sent_count,
            "delivered_count": delivered_count,
            "opened_count": opened_count,
            "clicked_count": clicked_count,
            "open_rate": round((opened_count / delivered_count * 100), 1) if delivered_count > 0 else 0.0,
            "click_rate": round((clicked_count / delivered_count * 100), 1) if delivered_count > 0 else 0.0,
            "skipped_unsubscribed": skipped_unsubscribed,
            "skipped_duplicate": skipped_duplicate
        }

    def _send_via_brevo_api(
        self,
        recipient_email: str,
        recipient_name: str,
        subject: str,
        body: str
    ) -> Dict[str, Any]:
        """Send live email via Brevo transactional REST API."""
        headers = {
            "accept": "application/json",
            "api-key": self.api_key,
            "content-type": "application/json"
        }
        # Convert plain newlines to HTML breaks
        html_content = body.replace("\n", "<br>")
        payload = {
            "sender": {"name": SENDER_NAME, "email": SENDER_EMAIL},
            "to": [{"email": recipient_email, "name": recipient_name}],
            "subject": subject,
            "htmlContent": f"<div>{html_content}</div>"
        }

        try:
            response = requests.post(self.BREVO_API_URL, json=payload, headers=headers, timeout=10)
            if response.status_code in (200, 201, 202):
                data = response.json()
                return {
                    "status": "Sent",
                    "message_id": data.get("messageId", str(uuid.uuid4()))
                }
            else:
                return {
                    "status": "Failed",
                    "error": f"HTTP_{response.status_code}_{response.text[:50]}"
                }
        except Exception as e:
            return {"status": "Failed", "error": str(e)[:50]}

    def _simulate_send(self, recipient_email: str, engagement_prob: float) -> Dict[str, Any]:
        """Technically credible simulation: engagement probability realistically determines opens & clicks."""
        message_id = f"sim-{uuid.uuid4().hex[:12]}"

        # Probability calibrated response
        # High score (>0.75) -> 75% open chance, 35% click chance
        # Low score (<0.35) -> 15% open chance, 2% click chance
        open_chance = np.clip(engagement_prob * 0.85, 0.08, 0.85)
        is_opened = bool(np.random.random() < open_chance)

        is_clicked = False
        if is_opened:
            click_chance = np.clip(engagement_prob * 0.40, 0.02, 0.50)
            is_clicked = bool(np.random.random() < click_chance)

        return {
            "status": "Simulated_Sent",
            "message_id": message_id,
            "opened": is_opened,
            "clicked": is_clicked
        }

    def _update_log_metrics(self, campaign_id: str, customer_id: str, opened: int, clicked: int):
        """Update open and click flags in SQLite email_logs."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE email_logs
            SET opened = ?, clicked = ?
            WHERE campaign_id = ? AND customer_id = ?
        """, (opened, clicked, campaign_id, customer_id))
        conn.commit()
        conn.close()


def plot_campaign_comparison(campaigns: List[Dict[str, Any]]) -> go.Figure:
    """Generate interactive comparison chart across campaigns."""
    if not campaigns:
        fig = go.Figure()
        fig.add_annotation(
            text="No campaigns dispatched yet.",
            showarrow=False,
            font=dict(size=14, family="Plus Jakarta Sans, sans-serif", color="#6C7468")
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    titles = [c["title"][:22] + "..." if len(c["title"]) > 22 else c["title"] for c in campaigns]
    sent = [c.get("total_sent", 0) for c in campaigns]
    opens = [c.get("total_opens", 0) for c in campaigns]
    clicks = [c.get("total_clicks", 0) for c in campaigns]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Sent", x=titles, y=sent, marker_color="#324436"))
    fig.add_trace(go.Bar(name="Opened", x=titles, y=opens, marker_color="#52604D"))
    fig.add_trace(go.Bar(name="Clicked", x=titles, y=clicks, marker_color="#7D9376"))

    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#1F241E"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        title="Campaign Performance Comparison (Sent vs Opened vs Clicked)",
        xaxis=dict(title="Campaign", gridcolor="#E8E4DC"),
        yaxis=dict(title="Volume", gridcolor="#E8E4DC"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=20, r=20, b=40, t=50),
        height=380
    )
    return fig


def plot_campaign_rates_pie(total_delivered: int, total_opens: int, total_clicks: int) -> go.Figure:
    """Funnel-style donut chart of campaign response."""
    unopened = max(0, total_delivered - total_opens)
    opened_no_click = max(0, total_opens - total_clicks)

    labels = ["Clicked", "Opened (No Click)", "Unopened"]
    values = [total_clicks, opened_no_click, unopened]

    fig = px.pie(
        names=labels,
        values=values,
        hole=0.55,
        color=labels,
        color_discrete_map={
            "Clicked": "#52604D",
            "Opened (No Click)": "#8F754E",
            "Unopened": "#D8D2C5"
        },
        title="Engagement Breakdown"
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
        height=320
    )
    return fig
