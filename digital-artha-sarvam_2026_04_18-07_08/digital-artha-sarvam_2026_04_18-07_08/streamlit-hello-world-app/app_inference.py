import streamlit as st
import mlflow
import pandas as pd
from datetime import datetime
import os
import torch
import torch.nn as nn
import numpy as np

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Rakshak - Fraud & Loan Management",
    page_icon="🛡️",
    layout="wide"
)

# MLflow tracking
mlflow.set_tracking_uri("databricks")

# ==================== MODEL LOADING ====================

@st.cache_resource
def load_fraud_model():
    """Load fraud detection model from MLflow"""
    try:
        model_uri = "runs:/cabe6e4aac7f42bc80e46f3fa402e885/hybrid_fraud_model"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Failed to load fraud model: {str(e)}")
        return None

@st.cache_resource
def load_loan_model():
    """Load credit eligibility model from Unity Catalog"""
    try:
        model_uri = "models:/workspace.fraud_detection.credit_eligibility_classifier/2"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Failed to load loan model: {str(e)}")
        return None

# ==================== HELPER FUNCTIONS ====================

def predict_fraud(transaction_data, model):
    """Predict fraud using loaded model"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Predict
        prediction = model.predict(df)
        
        # Parse prediction (adjust based on actual model output format)
        if isinstance(prediction, pd.DataFrame):
            fraud_prob = float(prediction["fraud_probability"].iloc[0])
            fraud_pred = int(prediction["fraud_prediction"].iloc[0])
        elif isinstance(prediction, np.ndarray):
            fraud_prob = float(prediction[0][0] if len(prediction[0]) > 0 else prediction[0])
            fraud_pred = int(fraud_prob > 0.1)  # Using threshold 0.1
        else:
            fraud_prob = float(prediction)
            fraud_pred = int(fraud_prob > 0.1)
        
        return {
            "success": True,
            "data": {
                "fraud_probability": fraud_prob,
                "fraud_prediction": fraud_pred
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def predict_loan_eligibility(applicant_data, model):
    """Predict loan eligibility using loaded model"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([applicant_data])
        
        # Predict
        prediction = model.predict(df)
        
        # Parse prediction
        if isinstance(prediction, pd.DataFrame):
            result = prediction.to_dict(orient="records")[0]
        elif isinstance(prediction, np.ndarray):
            result = {"prediction": float(prediction[0])}
        else:
            result = {"prediction": prediction}
        
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_risk_color(probability):
    """Return color based on fraud probability"""
    if probability >= 0.7:
        return "🔴", "#ff4444"
    elif probability >= 0.4:
        return "🟡", "#ffaa00"
    else:
        return "🟢", "#44ff44"

# ==================== MAIN APP ====================

st.title("🛡️ Rakshak - Fraud Detection & Loan Management System")
st.markdown("**Real-time ML-powered Financial Services (In-App Inference)**")
st.markdown("---")

# Load models
with st.spinner("Loading ML models..."):
    fraud_model = load_fraud_model()
    loan_model = load_loan_model()

# Sidebar for model status
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    st.subheader("📊 Model Status")
    
    if fraud_model:
        st.success("✅ Fraud Detection Model: Loaded")
    else:
        st.error("❌ Fraud Detection Model: Failed")
    
    if loan_model:
        st.success("✅ Credit Eligibility Model: Loaded")
    else:
        st.error("❌ Credit Eligibility Model: Failed")
    
    st.markdown("---")
    
    st.subheader("ℹ️ System Info")
    st.text("Inference: In-App (No endpoints)")
    st.text("Models: 2 Loaded")
    st.text("Status: Production")

# ==================== TAB 1: FRAUD DETECTION ====================

tab1, tab2, tab3 = st.tabs(["🚨 Fraud Detection", "💰 Credit Eligibility", "📚 Documentation"])

with tab1:
    st.header("🚨 UPI Transaction Fraud Detection")
    st.markdown("Analyze real-time UPI transactions for potential fraud using our hybrid XGBoost + DQN model.")
    
    if not fraud_model:
        st.error("❌ Fraud detection model not loaded. Cannot perform predictions.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Transaction Details")
            
            with st.form("fraud_form"):
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown("**💰 Transaction Info**")
                    amount_inr = st.number_input("Amount (₹)", min_value=0.0, value=3674.69, step=100.0)
                    txn_type_enc = st.selectbox("Transaction Type", 
                        options=[0, 1, 2, 3], 
                        format_func=lambda x: ["P2P", "P2M", "Bill Payment", "Recharge"][x],
                        index=3)
                    cat_enc = st.selectbox("Merchant Category", 
                        options=[0, 1, 2], 
                        format_func=lambda x: ["Food & Dining", "Retail", "Services"][x],
                        index=2)
                    device_enc = st.selectbox("Device Type", 
                        options=[0, 1, 2], 
                        format_func=lambda x: ["Android", "iOS", "Web"][x],
                        index=2)
                
                with c2:
                    st.markdown("**⏰ Time Details**")
                    hour_of_day = st.slider("Hour of Day", 0, 23, 3)
                    day_of_week = st.slider("Day of Week", 0, 6, 0)
                    is_weekend = st.checkbox("Is Weekend", value=False)
                    network_enc = st.selectbox("Network Type", 
                        options=[0, 1, 2], 
                        format_func=lambda x: ["WiFi", "4G", "5G"][x],
                        index=0)
                
                with c3:
                    st.markdown("**👤 User Details**")
                    sender_age_enc = st.selectbox("Sender Age Group", 
                        options=[0, 1, 2, 3, 4],
                        format_func=lambda x: ["18-25", "26-35", "36-45", "46-60", "60+"][x],
                        index=1)
                    receiver_age_enc = st.selectbox("Receiver Age Group", 
                        options=[0, 1, 2, 3, 4],
                        format_func=lambda x: ["18-25", "26-35", "36-45", "46-60", "60+"][x],
                        index=0)
                    sender_avg_amount_prev = st.number_input("Sender Avg Amount (₹)", value=4303.93, step=100.0)
                    sender_txn_count_prev = st.number_input("Sender Total Transactions", value=77, step=1, min_value=0)
                
                submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True, type="primary")
                
                if submitted:
                    # Prepare transaction data
                    amount_norm = amount_inr / 50000.0
                    amount_vs_sender_mean = amount_inr / sender_avg_amount_prev if sender_avg_amount_prev > 0 else 0
                    odd_hour_flag = 1 if hour_of_day < 6 or hour_of_day > 22 else 0
                    high_amount_flag = 1 if amount_inr > 10000 else 0
                    
                    transaction = {
                        'day_of_week': day_of_week,
                        'device_enc': device_enc,
                        'amount_norm': amount_norm,
                        'hour_of_day': hour_of_day,
                        'txn_velocity': 0.0,
                        'odd_hour_flag': odd_hour_flag,
                        'device_network_interaction': device_enc * 100 + network_enc,
                        'sender_avg_amount_prev': sender_avg_amount_prev,
                        'amount_weekend': amount_inr * (1 if is_weekend else 0),
                        'fuel_large_flag': 1 if cat_enc == 2 and amount_inr > 5000 else 0,
                        'amount_vs_sender_mean': amount_vs_sender_mean,
                        'high_amount_odd_hour': 1 if high_amount_flag and odd_hour_flag else 0,
                        'cat_device_interaction': cat_enc * 100 + device_enc,
                        'same_bank_flag': 0,
                        'sender_age_enc': sender_age_enc,
                        'weekend_high_spend': 1 if is_weekend and amount_inr > 5000 else 0,
                        'txn_type_enc': txn_type_enc,
                        'high_amount_flag': high_amount_flag,
                        'amount_hour_interaction': amount_inr * hour_of_day / 1000.0,
                        'sender_txn_count_prev': int(sender_txn_count_prev),
                        'amount_inr': amount_inr,
                        'is_weekend': 1 if is_weekend else 0,
                        'receiver_age_enc': receiver_age_enc,
                        'network_enc': network_enc,
                        'sender_max_amount_prev': sender_avg_amount_prev * 1.5,
                        'amount_ratio_deviation': abs(amount_vs_sender_mean - 1.0),
                        'cat_enc': cat_enc
                    }
                    
                    with st.spinner("🔍 Analyzing transaction..."):
                        result = predict_fraud(transaction, fraud_model)
                        
                        if result["success"]:
                            fraud_prob = result["data"]["fraud_probability"]
                            fraud_pred = result["data"]["fraud_prediction"]
                            
                            st.session_state['fraud_result'] = {
                                'probability': fraud_prob,
                                'is_fraud': fraud_pred == 1,
                                'amount': amount_inr,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
        
        with col2:
            st.subheader("Analysis Results")
            
            if 'fraud_result' in st.session_state:
                result = st.session_state['fraud_result']
                
                icon, color = get_risk_color(result['probability'])
                st.markdown(f"### {icon} Risk Assessment")
                
                prob_percentage = result['probability'] * 100
                st.progress(result['probability'])
                st.markdown(f"<h1 style='text-align: center; color: {color}'>{prob_percentage:.1f}%</h1>", 
                           unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: gray'>Fraud Probability</p>", 
                           unsafe_allow_html=True)
                
                st.markdown("---")
                
                if result['is_fraud']:
                    st.error("### 🚨 FRAUD DETECTED!")
                    st.markdown("""
                    **Recommended Actions:**
                    - 🔒 Block transaction immediately
                    - 📞 Contact sender for verification
                    - 🚨 Flag account for investigation
                    """)
                else:
                    st.success("### ✅ Transaction Appears Safe")
                    st.markdown("""
                    **Next Steps:**
                    - ✅ Process transaction normally
                    - 📊 Continue monitoring
                    """)
                
                st.markdown("---")
                st.markdown("#### 📋 Transaction Summary")
                st.write(f"**Amount:** ₹{result['amount']:,.2f}")
                st.write(f"**Analyzed At:** {result['timestamp']}")
                
                if st.button("🔄 Clear Results", use_container_width=True):
                    del st.session_state['fraud_result']
                    st.rerun()
            else:
                st.info("👈 Fill in transaction details to see results")

# ==================== TAB 2: CREDIT ELIGIBILITY ====================

with tab2:
    st.header("💰 Credit Eligibility Assessment")
    st.markdown("Banking behavior-based credit eligibility powered by ML")
    
    if not loan_model:
        st.error("❌ Credit eligibility model not loaded. Cannot perform predictions.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Banking Activity Metrics")
            
            with st.form("loan_form"):
                c1, c2 = st.columns(2)
                
                with c1:
                    monthly_inflow_avg = st.number_input("Avg Monthly Inflow (₹)", 
                        min_value=0.0, value=50000.0, step=1000.0)
                    avg_transaction_amount = st.number_input("Avg Transaction Amount (₹)", 
                        min_value=0.0, value=2500.0, step=100.0)
                    max_transaction = st.number_input("Max Transaction (₹)", 
                        min_value=0.0, value=25000.0, step=1000.0)
                    active_days = st.slider("Active Days per Month", 0, 31, 20)
                
                with c2:
                    transaction_velocity = st.number_input("Transaction Velocity", 
                        min_value=0.0, value=5.5, step=0.1)
                    transaction_amount_volatility = st.slider("Amount Volatility", 
                        0.0, 1.0, 0.35, 0.01)
                    bounce_rate = st.slider("Bounce Rate (%)", 
                        0.0, 100.0, 2.5, 0.5)
                    credit_debit_ratio = st.slider("Credit/Debit Ratio", 
                        0.0, 5.0, 1.2, 0.1)
                
                st.markdown("**📈 Transaction Patterns**")
                c3, c4 = st.columns(2)
                
                with c3:
                    large_txn_ratio = st.slider("Large Transaction Ratio", 0.0, 1.0, 0.15, 0.01)
                    small_txn_ratio = st.slider("Small Transaction Ratio", 0.0, 1.0, 0.45, 0.01)
                
                with c4:
                    weekend_txn_ratio = st.slider("Weekend Transaction Ratio", 0.0, 1.0, 0.28, 0.01)
                    night_txn_ratio = st.slider("Night Transaction Ratio", 0.0, 1.0, 0.12, 0.01)
                
                submitted_loan = st.form_submit_button("🔍 Check Eligibility", 
                    use_container_width=True, type="primary")
                
                if submitted_loan:
                    applicant_data = {
                        'monthly_inflow_avg': monthly_inflow_avg,
                        'transaction_velocity': transaction_velocity,
                        'bounce_rate': bounce_rate,
                        'avg_transaction_amount': avg_transaction_amount,
                        'transaction_amount_volatility': transaction_amount_volatility,
                        'large_txn_ratio': large_txn_ratio,
                        'max_transaction': max_transaction,
                        'weekend_txn_ratio': weekend_txn_ratio,
                        'night_txn_ratio': night_txn_ratio,
                        'credit_debit_ratio': credit_debit_ratio,
                        'active_days': float(active_days),
                        'small_txn_ratio': small_txn_ratio
                    }
                    
                    with st.spinner("🔍 Analyzing banking behavior..."):
                        result = predict_loan_eligibility(applicant_data, loan_model)
                        
                        if result["success"]:
                            st.session_state['loan_result'] = {
                                'raw_response': result["data"],
                                'monthly_inflow': monthly_inflow_avg,
                                'bounce_rate': bounce_rate,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
        
        with col2:
            st.subheader("📊 Eligibility Results")
            
            if 'loan_result' in st.session_state:
                result = st.session_state['loan_result']
                
                st.success("### ✅ Analysis Complete")
                st.markdown("#### 🎯 Model Response")
                st.json(result['raw_response'])
                
                st.markdown("---")
                st.markdown("#### 📋 Banking Profile")
                st.write(f"**Avg Monthly Inflow:** ₹{result['monthly_inflow']:,.2f}")
                st.write(f"**Bounce Rate:** {result['bounce_rate']:.1f}%")
                st.write(f"**Analyzed At:** {result['timestamp']}")
                
                if st.button("🔄 New Analysis", use_container_width=True):
                    del st.session_state['loan_result']
                    st.rerun()
            else:
                st.info("👈 Fill in banking metrics to see results")

# ==================== TAB 3: DOCUMENTATION ====================

with tab3:
    st.header("📚 Model Documentation")
    
    st.markdown("## 🎯 Inference Method")
    st.info("✅ **In-App Inference:** Models loaded directly from MLflow - No Model Serving endpoints used")
    
    st.markdown("### Fraud Detection Model")
    st.code("""
Model URI: runs:/cabe6e4aac7f42bc80e46f3fa402e885/hybrid_fraud_model
Type: Hybrid (XGBoost + DQN)
Features: 27 transaction features
Output: fraud_probability, fraud_prediction
    """)
    
    st.markdown("### Credit Eligibility Model")
    st.code("""
Model URI: models:/workspace.fraud_detection.credit_eligibility_classifier/2
Type: Banking Behavior Classifier
Features: 12 banking metrics
Output: eligibility score/prediction
    """)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h4>🛡️ Rakshak Financial Services Platform</h4>
    <p style='color: gray; font-size: 0.9em'>
        In-App ML Inference | No Dedicated Endpoints | Hackathon Compliant
    </p>
</div>
""", unsafe_allow_html=True)
