"""
Configuration settings for Streamlit application.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# Model file paths
NO_SHOW_MODEL_PATH = MODELS_DIR / 'no_show_classifier.joblib'
DEMAND_FORECASTER_PATH = MODELS_DIR / 'demand_forecaster.joblib'
PREPROCESSOR_PATH = MODELS_DIR / 'preprocessor.joblib'

# Default settings
DEFAULT_SPECIALTY = 'All'
DEFAULT_LOCATION = 'All'
DEFAULT_FORECAST_DAYS = 30

# Risk thresholds for no-show prediction
RISK_THRESHOLDS = {
    'Low': (0, 0.33),
    'Medium': (0.33, 0.67),
    'High': (0.67, 1.0)
}

# Streamlit page configuration
PAGE_TITLE = "Medical Appointment Management System"
PAGE_ICON = "🏥"
LAYOUT = "wide"

# Color scheme
COLORS = {
    'low_risk': '#2ecc71',      # Green
    'medium_risk': '#f39c12',   # Orange
    'high_risk': '#e74c3c',     # Red
    'neutral': '#3498db',       # Blue
    'background': '#ecf0f1',    # Light gray
}

# Feature configurations
PATIENT_FEATURES = [
    'gender', 'age', 'disability', 'needs_companion',
    'Hypertension', 'Diabetes', 'Alcoholism', 'Handcap', 'Scholarship'
]

APPOINTMENT_FEATURES = [
    'specialty', 'appointment_shift', 'place', 'SMSreceived'
]

WEATHER_FEATURES = [
    'avg_temp', 'max_temp', 'rain', 'heat_intensity', 'rain_intensity',
    'rainy_day_before', 'storm_day_before'
]

# Specialty list
SPECIALTIES = [
    'All',
    'Physiotherapy', 'Psychotherapy', 'Speech Therapy', 'Occupational Therapy',
    'Pediatric Physiotherapy', 'Adult Physical Therapy'
]

LOCATIONS = ['All']

# Shifts
SHIFTS = ['Morning', 'Afternoon']

# Genders
GENDERS = ['Male', 'Female']
