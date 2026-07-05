"""
config.py
Central configuration for the Clinical Trial Site Performance Advisor
"""

from dataclasses import dataclass

APP_NAME = "Clinical Trial Site Performance Advisor"
APP_VERSION = "2.0"
PAGE_TITLE = APP_NAME
PAGE_ICON = "🧬"
LAYOUT = "wide"

HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
HF_API_TIMEOUT = 180
MAX_NEW_TOKENS = 600
TEMPERATURE = 0.2
TOP_P = 0.95
REPETITION_PENALTY = 1.05
WAIT_FOR_MODEL = True

HEALTH_THRESHOLDS = {
    "Excellent": 90,
    "Good": 75,
    "Fair": 60,
    "Poor": 0,
}

ENROLLMENT_THRESHOLDS = {
    "excellent": 0.95,
    "good": 0.80,
    "fair": 0.60,
}

SCREEN_FAILURE_THRESHOLDS = {
    "low": 0.10,
    "medium": 0.20,
    "high": 0.30,
}

TIMELINE_DELAY_THRESHOLDS = {
    "low": 5,
    "medium": 15,
    "high": 30,
}

RISK_THRESHOLDS = {
    "low": 25,
    "medium": 50,
    "high": 75,
}

HEALTH_COLORS = {
    "Excellent": "#2E8B57",
    "Good": "#3CB371",
    "Fair": "#F4C430",
    "Poor": "#DC143C",
}

HEALTH_ICONS = {
    "Excellent": "🏆",
    "Good": "🟢",
    "Fair": "🟡",
    "Poor": "🔴",
}

SUPPORTED_INTENTS = [
    "general_analysis",
    "explain_poor",
    "explain_fair",
    "explain_good",
    "explain_excellent",
    "improve_poor",
    "improve_good",
    "compare_sites",
    "executive_summary",
    "timeline_analysis",
    "screen_failure",
    "recruitment",
    "budget",
]

REQUIRED_COLUMNS = [
    "site_id",
    "site_name",
    "site_health",
]

DEFAULT_RESPONSE_SECTIONS = [
    "Executive Summary",
    "Operational Findings",
    "Clinical Findings",
    "Business Impact",
    "Recommendations",
    "Expected Outcome",
    "Confidence",
]

@dataclass(frozen=True)
class ModelSettings:
    model_name: str = HF_MODEL
    timeout: int = HF_API_TIMEOUT
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_NEW_TOKENS
    top_p: float = TOP_P
    repetition_penalty: float = REPETITION_PENALTY

MODEL_SETTINGS = ModelSettings()
