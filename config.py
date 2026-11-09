"""Configuration settings and environment variable management for IntelliMail AI."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "intellimail.db"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# API Keys & Credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "").strip()
SENDER_NAME = os.getenv("SENDER_NAME", "IntelliMail AI Marketing")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "marketing@example.com")

# Model configuration
ENGAGEMENT_MODEL_PATH = MODELS_DIR / "engagement_model.joblib"
SCALER_MODEL_PATH = MODELS_DIR / "rfm_scaler.joblib"
KMEANS_MODEL_PATH = MODELS_DIR / "rfm_kmeans.joblib"

# UI Theme & Branding
APP_TITLE = "IntelliMail AI"
APP_SUBTITLE = "AI-Powered Email Marketing Automation System"
APP_TAGLINE = "Intelligent Marketing Automation Platform"

BRAND_COLORS = {
    "primary": "#52604D",      # Muted Olive
    "secondary": "#3E4C38",    # Deep Olive
    "accent": "#73866C",       # Soft Sage
    "highlight": "#8F754E",    # Warm Ochre
    "dark_bg": "#1C211A",      # Ink Charcoal
    "light_bg": "#FAF8F5",     # Warm Ivory
    "card_bg": "#FFFFFF",      # Crisp White
    "border": "#E8E4DC",       # Subtle Warm Border
    "success": "#46634B",      # Muted Forest
    "warning": "#8F754E",      # Warm Ochre
    "danger": "#935353",       # Muted Terracotta
}

AVAILABLE_TONES = [
    "Professional",
    "Persuasive",
    "Friendly",
    "Urgent",
    "Promotional",
    "Educational",
    "Empathetic"
]

CUSTOMER_INTERESTS = [
    "Artificial Intelligence",
    "Cloud Computing",
    "Cybersecurity",
    "Data Analytics",
    "E-Commerce",
    "SaaS Tools",
    "Digital Marketing",
    "Finance & Fintech"
]

SEGMENT_NAMES = {
    0: "Champions (High Value)",
    1: "Loyal Customers",
    2: "Promising Prospects",
    3: "At-Risk / Inactive"
}
