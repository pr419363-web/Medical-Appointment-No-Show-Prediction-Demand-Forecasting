"""
Main Streamlit Application for Medical Appointment Management System.
Provides two modules: No-Show Predictor and Demand Forecaster.

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import PAGE_TITLE, PAGE_ICON, LAYOUT, COLORS

# Configure page
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state='expanded'
)

# Custom CSS for styling
st.markdown("""
    <style>
    .header-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-low {
        color: #2ecc71;
        font-weight: bold;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("### CER Analytics")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "🔍 Select Module",
    ["Home", "📋 No-Show Predictor", "📊 Demand Forecaster", "ℹ️ About"],
    help="Choose the feature you want to use"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Quick Links")
st.sidebar.info(
    "**System Features:**\n"
    "• No-show model evaluation\n"
    "• Daily demand backtesting\n"
    "• Input screens are previews\n"
    "• No live predictions served"
)

# ============================================================================
# HOME PAGE
# ============================================================================
if app_mode == "Home":
    st.markdown('<p class="header-text">🏥 Medical Appointment Management System</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Medical Appointment No-Show Prediction & Demand Forecasting System**
    developed for the University of Vale do Itajaí Center of Specialization in Physical 
    and Intellectual Rehabilitation (CER).
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 No-Show Predictor")
        st.markdown("""
        Identify patients at risk of missing their appointments based on:
        - Patient demographics
        - Appointment characteristics
        - Health conditions
        - Weather conditions
        
        **Intended use**: Evaluate targeted outreach after a validated model is available
        """)
    
    with col2:
        st.markdown("### 📈 Demand Forecaster")
        st.markdown("""
        Forecast daily appointment volumes to optimize:
        - Staff scheduling
        - Resource allocation
        - Capacity planning
        - Specialty-specific staffing
        
        **Intended use**: Assess staffing plans after forecast performance improves
        """)
    
    st.markdown("---")
    
    # Key metrics overview
    st.markdown("### 📊 System Overview")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Historical No-Show Rate", "31.8%")
    
    with metric_col2:
        st.metric("Classifier F1 (Baseline)", "0.56")
    
    with metric_col3:
        st.metric("Forecast MAPE (Baseline)", "5,008.8%")
    
    with metric_col4:
        st.metric("Distinct Place Strings", "26,289")
    
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    
    st.markdown("""
    1. **No-Show Predictor**: Enter sample details in the preview form. No individual score is served.
    
    2. **Demand Forecaster**: Review the preview controls. No future forecast is served.

    Run `python src/evaluate_baselines.py` to regenerate the held-out evaluation results and plots.
    """)

# ============================================================================
# NO-SHOW PREDICTOR PAGE
# ============================================================================
elif app_mode == "📋 No-Show Predictor":
    st.markdown('<p class="header-text">📋 No-Show Risk Predictor</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Preview the patient and appointment fields. Individual risk scores are not served
    until a trained inference model is available.
    """)
    
    # Create two columns for input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Patient Information")
        
        patient_gender = st.selectbox("Gender", ["Male", "Female"])
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=40)
        has_disability = st.checkbox("Has Disability")
        needs_companion = st.checkbox("Needs Companion")
        
        st.markdown("### Health Conditions")
        has_hypertension = st.checkbox("Hypertension")
        has_diabetes = st.checkbox("Diabetes")
        has_alcoholism = st.checkbox("Alcoholism")
        has_handcap = st.checkbox("Physical Handicap")
        on_scholarship = st.checkbox("On Scholarship")
    
    with col2:
        st.markdown("### Appointment Details")
        
        specialty = st.selectbox(
            "Specialty",
            ["Physiotherapy", "Psychotherapy", "Speech Therapy", 
             "Occupational Therapy", "Other"]
        )
        
        location = st.text_input("Location label", value="ITAPEMA")
        
        appointment_shift = st.selectbox("Appointment Shift", ["Morning", "Afternoon"])
        sms_received = st.checkbox("SMS Reminder Received")
        
        st.markdown("### Weather Conditions")
        avg_temp = st.slider("Average Temperature (°C)", 15, 40, 25)
        max_temp = st.slider("Max Temperature (°C)", 20, 45, 30)
        rainfall = st.slider("Rainfall (mm)", 0, 100, 5)
    
    # Predict button
    st.markdown("---")
    
    if st.button("🔮 Predict No-Show Risk", use_container_width=True):
        st.warning(
            "No trained inference model is available yet. The evaluation baseline is "
            "not packaged for individual predictions."
        )

# ============================================================================
# DEMAND FORECASTER PAGE
# ============================================================================
elif app_mode == "📊 Demand Forecaster":
    st.markdown('<p class="header-text">📊 Appointment Demand Forecaster</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Preview the forecast controls. Future demand predictions are not served until a
    trained forecasting model is available.
    """)
    
    # Forecast parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_days = st.number_input("Forecast Period (days)", 7, 90, 30)
    
    with col2:
        specialty_filter = st.selectbox(
            "Specialty",
            ["All Specialties", "Physiotherapy", "Psychotherapy", 
             "Speech Therapy", "Occupational Therapy"]
        )
    
    with col3:
        location_filter = st.text_input("Location label", value="All locations")
    
    # Generate forecast
    if st.button("🚀 Generate Forecast", use_container_width=True):
        st.warning(
            "No trained forecasting model is available yet. Synthetic demo forecasts "
            "have been removed so this screen does not imply a validated prediction."
        )

# ============================================================================
# ABOUT PAGE
# ============================================================================
elif app_mode == "ℹ️ About":
    st.markdown('<p class="header-text">ℹ️ About This System</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### System Purpose
    
    This application helps the University of Vale do Itajaí Center of Specialization in 
    Physical and Intellectual Rehabilitation (CER) address two critical operational challenges:
    
    1. **High No-Show Rates (31.8%)**: The clinic experiences significantly higher patient 
       no-show rates than industry standard (10-20%), leading to:
       - Revenue loss from unused specialist capacity
       - Inefficient staff scheduling
       - Reduced service quality
    
    2. **Unpredictable Demand**: Without demand forecasting, the clinic struggles to:
       - Optimize specialist scheduling
    - Validate location labels before geographic resource planning
       - Plan for seasonal variations
    
    ### Solution Overview
    
    **Module 1: No-Show Risk Predictor**
    - The Streamlit form is a UI preview; it does not serve individual predictions.
    - The evaluated Random Forest baseline achieved F1 = 0.56 and ROC-AUC = 0.77.
    
    **Module 2: Demand Forecaster**
    - The Streamlit controls are a UI preview; they do not serve future forecasts.
    - The overall daily-demand baseline achieved MAPE = 5,008.82% and R² = -0.0033 on a chronological holdout.
    
    ### Key Data Sources
    
    - **Patient Data**: Demographics, health conditions, appointment history
    - **Appointment Data**: Specialty, date, time, location
    - **Weather Data**: Temperature, rainfall, intensity indicators
    - **Historical Patterns**: No-show rates, demand trends
    
    ### Impact Status
    
    No intervention or operational efficiency improvement has been measured. The README describes an illustrative reminder scenario; it must be tested in a controlled pilot before being treated as an outcome.
    
    ### Technical Stack
    
    - **Data Processing**: Python, Pandas, NumPy
    - **Machine Learning**: Scikit-learn, XGBoost, LightGBM
    - **Time Series**: ARIMA, Prophet, LSTM
    - **Application**: Streamlit
    - **Visualization**: Plotly, Matplotlib, Seaborn
    
    ### Data Privacy & Security
    
    ✅ All patient data is processed locally  
    ✅ No data is transmitted to external servers  
    ✅ Models are trained on historical aggregate patterns  
    ✅ Individual predictions are not stored  
    
    ### Support & Documentation
    
    For technical support or questions about the system:
    - Review the project README file
    - Check the Jupyter notebooks for detailed explanations
    - Contact the development team for deployment questions
    
    ---
    
    **System Version**: 1.0  
    **Last Updated**: June 2026  
    **Status**: ✅ Production Ready
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>
**Medical Appointment Management System**  
Version 1.0 | June 2026  
University of Vale do Itajaí CER
</small>
""", unsafe_allow_html=True)
