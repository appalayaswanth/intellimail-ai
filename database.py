"""SQLite database layer for IntelliMail AI.
Handles customers, campaigns, send logs, unsubscribe lists, and duplicate prevention.
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from config import DB_PATH


def get_connection():
    """Create a connection to the SQLite database with WAL mode and row factory enabled."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables with complete schema, indices, and constraints."""
    conn = get_connection()
    cursor = conn.cursor()

    # Customers Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            interest TEXT NOT NULL,
            company TEXT DEFAULT 'Enterprise',
            recency INTEGER NOT NULL,
            frequency INTEGER NOT NULL,
            monetary REAL NOT NULL,
            previous_open_rate REAL NOT NULL,
            previous_click_rate REAL NOT NULL,
            segment_id INTEGER DEFAULT -1,
            segment_name TEXT DEFAULT 'Unsegmented',
            engagement_score REAL DEFAULT 0.0,
            is_subscribed INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Campaigns Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            subject TEXT NOT NULL,
            body TEXT NOT NULL,
            target_segment TEXT DEFAULT 'All',
            target_min_prob REAL DEFAULT 0.0,
            status TEXT DEFAULT 'Draft',
            scheduled_at TEXT,
            sent_at TEXT,
            total_recipients INTEGER DEFAULT 0,
            total_sent INTEGER DEFAULT 0,
            total_delivered INTEGER DEFAULT 0,
            total_opens INTEGER DEFAULT 0,
            total_clicks INTEGER DEFAULT 0,
            is_approved INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Email Send Logs (Tracks individual emails for duplicate prevention & audit)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS email_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            recipient_email TEXT NOT NULL,
            subject TEXT NOT NULL,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            delivery_status TEXT NOT NULL,
            message_id TEXT,
            opened INTEGER DEFAULT 0,
            clicked INTEGER DEFAULT 0,
            UNIQUE(campaign_id, customer_id)
        )
    """)

    # Email Templates Table (For saving AI-generated drafts)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS email_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            subject TEXT NOT NULL,
            body TEXT NOT NULL,
            tone TEXT NOT NULL,
            target_interest TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indices for fast lookup
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_segment ON customers(segment_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_campaign ON email_logs(campaign_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_recipient ON email_logs(recipient_email)")

    conn.commit()
    conn.close()


def upsert_customers_from_df(df: pd.DataFrame, replace_existing: bool = False) -> int:
    """Bulk insert/upsert customers into SQLite from a Pandas DataFrame.
    Uses INSERT OR REPLACE to prevent duplicate key constraint crashes.
    If replace_existing is True, clears the customers table first.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if replace_existing:
        cursor.execute("DELETE FROM customers")

    count = 0
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO customers (
                customer_id, name, email, interest, company,
                recency, frequency, monetary,
                previous_open_rate, previous_click_rate,
                segment_id, segment_name, engagement_score, is_subscribed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(row.get('customer_id', f"CUST-{count+1:05d}")),
            str(row.get('name', 'Valued Customer')),
            str(row.get('email', f"user{count+1}@example.com")).strip().lower(),
            str(row.get('interest', 'General')),
            str(row.get('company', 'Enterprise')),
            int(row.get('recency', 30)),
            int(row.get('frequency', 5)),
            float(row.get('monetary', 100.0)),
            float(row.get('previous_open_rate', 0.2)),
            float(row.get('previous_click_rate', 0.05)),
            int(row.get('segment_id', -1)),
            str(row.get('segment_name', 'Unsegmented')),
            float(row.get('engagement_score', 0.0)),
            int(row.get('is_subscribed', 1))
        ))
        count += 1

    conn.commit()
    conn.close()
    return count


def get_all_customers_df() -> pd.DataFrame:
    """Retrieve all customers as a DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM customers ORDER BY id ASC", conn)
    conn.close()
    return df


def get_customer_count() -> int:
    """Get total customer count in database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM customers")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def update_customer_segments(df: pd.DataFrame):
    """Update segment_id and segment_name for customers."""
    conn = get_connection()
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("""
            UPDATE customers
            SET segment_id = ?, segment_name = ?
            WHERE customer_id = ?
        """, (int(row['segment_id']), str(row['segment_name']), str(row['customer_id'])))
    conn.commit()
    conn.close()


def update_customer_engagement_scores(scores_dict: Dict[str, float]):
    """Update engagement_score for customers."""
    conn = get_connection()
    cursor = conn.cursor()
    for cust_id, score in scores_dict.items():
        cursor.execute("""
            UPDATE customers
            SET engagement_score = ?
            WHERE customer_id = ?
        """, (float(score), str(cust_id)))
    conn.commit()
    conn.close()


def unsubscribe_customer(email: str) -> bool:
    """Unsubscribe a recipient (respects consent requirement)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE customers SET is_subscribed = 0 WHERE email = ?", (email.strip().lower(),))
    rows = cursor.rowcount
    conn.commit()
    conn.close()
    return rows > 0


def resubscribe_customer(email: str) -> bool:
    """Resubscribe a recipient upon consent."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE customers SET is_subscribed = 1 WHERE email = ?", (email.strip().lower(),))
    rows = cursor.rowcount
    conn.commit()
    conn.close()
    return rows > 0


def is_already_sent(campaign_id: str, customer_id: str) -> bool:
    """Check if an email has already been sent to this recipient for this campaign."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM email_logs
        WHERE campaign_id = ? AND customer_id = ?
    """, (campaign_id, customer_id))
    exists = cursor.fetchone()[0] > 0
    conn.close()
    return exists


def save_campaign(campaign_data: Dict[str, Any]) -> str:
    """Create or update a campaign."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO campaigns (
            campaign_id, title, subject, body, target_segment,
            target_min_prob, status, scheduled_at, sent_at,
            total_recipients, total_sent, total_delivered,
            total_opens, total_clicks, is_approved
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(campaign_id) DO UPDATE SET
            title=excluded.title,
            subject=excluded.subject,
            body=excluded.body,
            target_segment=excluded.target_segment,
            target_min_prob=excluded.target_min_prob,
            status=excluded.status,
            scheduled_at=excluded.scheduled_at,
            sent_at=excluded.sent_at,
            total_recipients=excluded.total_recipients,
            total_sent=excluded.total_sent,
            total_delivered=excluded.total_delivered,
            total_opens=excluded.total_opens,
            total_clicks=excluded.total_clicks,
            is_approved=excluded.is_approved
    """, (
        campaign_data['campaign_id'],
        campaign_data['title'],
        campaign_data['subject'],
        campaign_data['body'],
        campaign_data.get('target_segment', 'All'),
        campaign_data.get('target_min_prob', 0.0),
        campaign_data.get('status', 'Draft'),
        campaign_data.get('scheduled_at'),
        campaign_data.get('sent_at'),
        campaign_data.get('total_recipients', 0),
        campaign_data.get('total_sent', 0),
        campaign_data.get('total_delivered', 0),
        campaign_data.get('total_opens', 0),
        campaign_data.get('total_clicks', 0),
        campaign_data.get('is_approved', 0)
    ))
    conn.commit()
    conn.close()
    return campaign_data['campaign_id']


def get_all_campaigns() -> List[Dict[str, Any]]:
    """Fetch all campaigns ordered by creation date."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM campaigns ORDER BY id DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_campaign(campaign_id: str) -> Optional[Dict[str, Any]]:
    """Fetch single campaign by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def log_email_send(campaign_id: str, customer_id: str, email: str, subject: str, status: str, message_id: str = None) -> bool:
    """Log an email sending attempt, returning True if logged or False if duplicate."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO email_logs (
                campaign_id, customer_id, recipient_email, subject, delivery_status, message_id
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (campaign_id, customer_id, email, subject, status, message_id))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        # Duplicate send prevented by unique constraint
        success = False
    finally:
        conn.close()
    return success


def get_campaign_logs(campaign_id: Optional[str] = None) -> pd.DataFrame:
    """Fetch email sending logs as DataFrame."""
    conn = get_connection()
    if campaign_id:
        query = "SELECT * FROM email_logs WHERE campaign_id = ? ORDER BY id DESC"
        df = pd.read_sql_query(query, conn, params=(campaign_id,))
    else:
        query = "SELECT * FROM email_logs ORDER BY id DESC LIMIT 500"
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def save_template(title: str, subject: str, body: str, tone: str, target_interest: str = "All"):
    """Save an email template to database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO email_templates (title, subject, body, tone, target_interest)
        VALUES (?, ?, ?, ?, ?)
    """, (title, subject, body, tone, target_interest))
    conn.commit()
    conn.close()


def get_templates() -> List[Dict[str, Any]]:
    """Retrieve saved email templates."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM email_templates ORDER BY id DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_dashboard_kpis() -> Dict[str, Any]:
    """Calculate aggregate KPIs for overview tab."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM customers")
    total_customers = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM customers WHERE is_subscribed = 1")
    subscribed_customers = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM campaigns")
    total_campaigns = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM email_logs")
    total_emails_sent = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM email_logs WHERE delivery_status IN ('Delivered', 'Sent', 'Simulated_Sent')")
    delivered_emails = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM email_logs WHERE opened = 1")
    opened_emails = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM email_logs WHERE clicked = 1")
    clicked_emails = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(engagement_score) FROM customers WHERE engagement_score > 0")
    avg_engagement = cursor.fetchone()[0] or 0.0

    conn.close()

    open_rate = (opened_emails / delivered_emails * 100) if delivered_emails > 0 else 0.0
    click_rate = (clicked_emails / delivered_emails * 100) if delivered_emails > 0 else 0.0

    return {
        "total_customers": total_customers,
        "subscribed_customers": subscribed_customers,
        "unsubscribed_customers": total_customers - subscribed_customers,
        "total_campaigns": total_campaigns,
        "total_emails_sent": total_emails_sent,
        "delivered_emails": delivered_emails,
        "opened_emails": opened_emails,
        "clicked_emails": clicked_emails,
        "open_rate": round(open_rate, 1),
        "click_rate": round(click_rate, 1),
        "avg_engagement": round(avg_engagement * 100, 1)
    }
