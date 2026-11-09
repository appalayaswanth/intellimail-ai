"""Comprehensive test suite for IntelliMail AI.
Validates all 4 core modules, safety guards, and persistence.
"""

import os
import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database import (
    init_db,
    upsert_customers_from_df,
    get_all_customers_df,
    get_customer_count,
    unsubscribe_customer,
    resubscribe_customer,
    is_already_sent,
    log_email_send,
    save_campaign,
    get_campaign,
    get_dashboard_kpis
)
from modules.ai_generator import AIEmailGenerator
from modules.segmentation import (
    generate_synthetic_customers,
    perform_rfm_clustering
)
from modules.engagement_model import (
    train_engagement_model,
    predict_customer_engagement,
    load_saved_model
)
from modules.email_automation import EmailAutomationService


class TestIntelliMailModules(unittest.TestCase):
    """Test suite covering the 4 core modules and safety requirements."""

    @classmethod
    def setUpClass(cls):
        """Initialize database schema once before tests."""
        init_db()

    def test_01_database_initialization_and_kpis(self):
        """Verify database tables and KPI computations function properly."""
        kpis = get_dashboard_kpis()
        self.assertIsInstance(kpis, dict)
        self.assertIn("total_customers", kpis)
        self.assertIn("open_rate", kpis)
        self.assertIn("click_rate", kpis)

    def test_02_ai_generator_demo_mode_and_personalization(self):
        """Verify Module 1 produces subject, body, and interpolates customer tags."""
        generator = AIEmailGenerator(api_key="")  # Force demo mode
        result = generator.generate_email(
            prompt="Announce Cloud Security Suite with 20% discount",
            tone="Urgent",
            target_interest="Cybersecurity",
            product_name="SecureCloud AI",
            call_to_action="Get Protected Now"
        )

        self.assertIn("subject", result)
        self.assertIn("body", result)
        self.assertTrue(len(result["subject"]) > 5)
        self.assertTrue(len(result["body"]) > 20)
        self.assertIn("AI", result["mode"])

        # Test token interpolation
        sample_cust = {
            "name": "Aarav Sharma",
            "interest": "Cybersecurity",
            "company": "Infosys",
            "email": "aarav@example.com"
        }
        personalized_body = generator.personalize_text(result["body"], sample_cust)
        self.assertNotIn("{name}", personalized_body)
        self.assertIn("Aarav Sharma", personalized_body)
        self.assertIn("Infosys", personalized_body)

    def test_03_customer_segmentation_rfm_kmeans(self):
        """Verify Module 2 generates 3,000 customers and segments with KMeans."""
        syn_df = generate_synthetic_customers(n=3000, seed=42)
        self.assertEqual(len(syn_df), 3000)
        self.assertIn("recency", syn_df.columns)
        self.assertIn("frequency", syn_df.columns)
        self.assertIn("monetary", syn_df.columns)

        clustered_df, scaler, kmeans, summary_df = perform_rfm_clustering(
            syn_df, n_clusters=4, save_models=True
        )

        self.assertEqual(len(clustered_df), 3000)
        self.assertIn("segment_id", clustered_df.columns)
        self.assertIn("segment_name", clustered_df.columns)
        self.assertEqual(len(summary_df), 4)

        # Upsert into database
        count = upsert_customers_from_df(clustered_df.head(200))
        self.assertEqual(count, 200)

    def test_04_engagement_prediction_logistic_regression(self):
        """Verify Module 3 trains CPU-based Logistic Regression and persists model."""
        syn_df = generate_synthetic_customers(n=1000, seed=42)
        train_results = train_engagement_model(syn_df, test_size=0.2, save_model=True)

        metrics = train_results["metrics"]
        self.assertGreater(metrics["accuracy"], 0.70)
        self.assertGreater(metrics["roc_auc"], 0.70)
        self.assertEqual(train_results["confusion_matrix"].shape, (2, 2))

        # Check joblib persistence
        loaded = load_saved_model()
        self.assertIsNotNone(loaded)

        # Predict batch
        probs = predict_customer_engagement(syn_df.head(10), pipeline=loaded)
        self.assertEqual(len(probs), 10)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_05_email_automation_safety_and_duplicate_prevention(self):
        """Verify Module 4 validates approval, consent, duplicate prevention, and simulation."""
        import uuid
        service = EmailAutomationService(api_key="")  # Simulation mode
        test_campaign_id = f"CAMP-TEST-{uuid.uuid4().hex[:6]}"

        # Test approval requirement
        test_df = pd.DataFrame([{
            "customer_id": f"TEST-{uuid.uuid4().hex[:4]}",
            "name": "Priya Patel",
            "email": f"priya.{uuid.uuid4().hex[:4]}@example.com",
            "interest": "AI",
            "company": "TechCorp",
            "engagement_score": 0.85,
            "is_subscribed": 1
        }])

        # Should block without explicit approval
        with self.assertRaises(PermissionError):
            service.dispatch_campaign(
                campaign_id=test_campaign_id,
                title="Test Blast",
                subject_template="Hello {name}",
                body_template="Hi {name}",
                recipients_df=test_df,
                is_approved=False
            )

        # Should succeed with explicit approval
        res = service.dispatch_campaign(
            campaign_id=test_campaign_id,
            title="Test Blast",
            subject_template="Hello {name}",
            body_template="Hi {name}",
            recipients_df=test_df,
            is_approved=True
        )
        self.assertEqual(res["sent_count"], 1)

        # Duplicate dispatch test: Re-sending same campaign to same user must be prevented!
        res_dup = service.dispatch_campaign(
            campaign_id=test_campaign_id,
            title="Test Blast",
            subject_template="Hello {name}",
            body_template="Hi {name}",
            recipients_df=test_df,
            is_approved=True
        )
        self.assertEqual(res_dup["skipped_duplicate"], 1)
        self.assertEqual(res_dup["sent_count"], 0)

        # Unsubscribe Consent Test
        unsubscribe_customer("priya.test@example.com")
        unsub_df = pd.DataFrame([{
            "customer_id": "TEST-002",
            "name": "Priya Patel",
            "email": "priya.test@example.com",
            "interest": "AI",
            "company": "TechCorp",
            "engagement_score": 0.85,
            "is_subscribed": 0  # Unsubscribed!
        }])
        res_unsub = service.dispatch_campaign(
            campaign_id="CAMP-TEST-2",
            title="Consent Test",
            subject_template="Hello {name}",
            body_template="Hi {name}",
            recipients_df=unsub_df,
            is_approved=True
        )
        self.assertEqual(res_unsub["skipped_unsubscribed"], 1)
        self.assertEqual(res_unsub["sent_count"], 0)


if __name__ == "__main__":
    unittest.main()
