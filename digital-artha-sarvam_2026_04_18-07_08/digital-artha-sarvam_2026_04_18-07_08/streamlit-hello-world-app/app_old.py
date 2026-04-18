import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import os

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Rakshak - Fraud & Loan Management",
    page_icon="🛡️",
    layout="wide"
)

# Your Databricks workspace URL and token
DATABRICKS_HOST = "https://dbc-a4d8c1e8-b4cf.cloud.databricks.com"
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

# Endpoint URLs
FRAUD_ENDPOINT = f"{DATABRICKS_HOST}/serving-endpoints/rakshak-fraud-api/invocations"
LOAN_ENDPOINT = f"{DATABRICKS_HOST}/serving-endpoints/loan-eligibility-api/invocations"

# Request headers
HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}

# ==================== HELPER FUNCTIONS ====================

def predict_fraud(transaction_data):
    """Call fraud detection endpoint"""
    try:
        payload = {
            "dataframe_records": [transaction_data]
        }
        response = requests.post(FRAUD_ENDPOINT, headers=HEADERS, json=payload, timeout=90)
        
        if response.status_code == 200:
            result = response.json()
            return {"success": True, "data": result["predictions"][0]}
        else:
            return {"success": False, "error": f"API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def predict_loan_eligibility(applicant_data):
    """Call loan eligibility endpoint"""
    try:
        payload = {
            "dataframe_records": [applicant_data]
        }
        response = requests.post(LOAN_ENDPOINT, headers=HEADERS, json=payload, timeout=90)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("predictions", [{}])[0]
            return {"success": True, "data": prediction}
        else:
            return {"success": False, "error": f"API Error: {response.status_code} - {response.text}"}
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

def check_endpoint_health(endpoint_name):
    """Check if endpoint is healthy"""
    try:
        response = requests.get(
            f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{endpoint_name}",
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            state = data.get("state", {}).get("ready", "UNKNOWN")
            return state == "READY"
        return False
    except:
        return False

# ==================== MAIN APP ====================

st.title("🛡️ Rakshak - Fraud Detection & Loan Management System")
st.markdown("**Real-time ML-powered Financial Services**")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # Token input
    if not DATABRICKS_TOKEN:
        st.warning("⚠️ Token not configured")
        token_input = st.text_input("Databricks Token", type="password", help="Enter your Databricks personal access token")
        if token_input:
            DATABRICKS_TOKEN = token_input
            HEADERS["Authorization"] = f"Bearer {DATABRICKS_TOKEN}"
    else:
        st.success("✅ Token Configured")
    
    st.markdown("---")
    
    # Check endpoint status
    st.subheader("📡 Endpoint Status")
    
    # Test fraud endpoint
    if DATABRICKS_TOKEN:
        fraud_healthy = check_endpoint_health("rakshak-fraud-api")
        if fraud_healthy:
            st.success("✅ Fraud Detection: Online")
        else:
            st.error("❌ Fraud Detection: Offline")
        
        # Test loan endpoint
        loan_healthy = check_endpoint_health("loan-eligibility-api")
        if loan_healthy:
            st.success("✅ Credit Eligibility: Online")
        else:
            st.error("❌ Credit Eligibility: Offline")
    else:
        st.info("🔑 Enter token to check status")
    
    st.markdown("---")
    
    # System info
    st.subheader("ℹ️ System Info")
    st.text(f"Host: {DATABRICKS_HOST}")
    st.text("Models: 2 Active")
    st.text("Status: Production")

# ==================== TAB 1: FRAUD DETECTION ====================

tab1, tab2, tab3 = st.tabs(["🚨 Fraud Detection", "💰 Credit Eligibility", "📚 Documentation"])

with tab1:
    st.header("🚨 UPI Transaction Fraud Detection")
    st.markdown("Analyze real-time UPI transactions for potential fraud using our hybrid XGBoost + DQN model.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Transaction Details")
        
        # Transaction input form
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
                day_of_week = st.slider("Day of Week", 0, 6, 0, 
                    help="0=Monday, 1=Tuesday, ..., 6=Sunday")
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
            
            # Auto-calculated features (shown as info)
            with st.expander("📊 Advanced Features (Auto-calculated)", expanded=False):
                amount_norm = amount_inr / 50000.0
                amount_vs_sender_mean = amount_inr / sender_avg_amount_prev if sender_avg_amount_prev > 0 else 0
                odd_hour_flag = 1 if hour_of_day < 6 or hour_of_day > 22 else 0
                high_amount_flag = 1 if amount_inr > 10000 else 0
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Amount Normalized", f"{amount_norm:.4f}")
                    st.metric("vs Sender Mean", f"{amount_vs_sender_mean:.4f}")
                with col_b:
                    st.metric("Odd Hour Flag", "Yes" if odd_hour_flag else "No")
                    st.metric("High Amount Flag", "Yes" if high_amount_flag else "No")
            
            submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True, type="primary")
            
            if submitted:
                if not DATABRICKS_TOKEN:
                    st.error("❌ Please enter your Databricks token in the sidebar first!")
                else:
                    # Prepare transaction data with all 27 features
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
                    
                    with st.spinner("🔍 Analyzing transaction with ML model..."):
                        result = predict_fraud(transaction)
                        
                        if result["success"]:
                            fraud_prob = result["data"].get("fraud_probability", 0.0)
                            fraud_pred = result["data"].get("fraud_prediction", 0)
                            
                            # Store in session state
                            st.session_state['fraud_result'] = {
                                'probability': fraud_prob,
                                'is_fraud': fraud_pred == 1,
                                'amount': amount_inr,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'transaction': transaction
                            }
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
                            st.info("💡 Check your token and endpoint status in the sidebar")
    
    with col2:
        st.subheader("Analysis Results")
        
        if 'fraud_result' in st.session_state:
            result = st.session_state['fraud_result']
            
            # Risk indicator
            icon, color = get_risk_color(result['probability'])
            st.markdown(f"### {icon} Risk Assessment")
            
            # Large probability display
            prob_percentage = result['probability'] * 100
            st.progress(result['probability'])
            st.markdown(f"<h1 style='text-align: center; color: {color}'>{prob_percentage:.1f}%</h1>", 
                       unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: gray'>Fraud Probability</p>", 
                       unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Verdict
            if result['is_fraud']:
                st.error("### 🚨 FRAUD DETECTED!")
                st.warning("**⚠️ Immediate Action Required**")
                st.markdown("""
                **Recommended Actions:**
                - 🔒 Block transaction immediately
                - 📞 Contact sender for verification
                - 🚨 Flag account for investigation
                - 📝 Document incident details
                """)
            else:
                st.success("### ✅ Transaction Appears Safe")
                st.info("**Action: Approve for processing**")
                st.markdown("""
                **Next Steps:**
                - ✅ Process transaction normally
                - 📊 Continue monitoring user activity
                - 💾 Log for future analysis
                """)
            
            st.markdown("---")
            
            # Transaction summary
            st.markdown("#### 📋 Transaction Summary")
            st.write(f"**Amount:** ₹{result['amount']:,.2f}")
            st.write(f"**Analyzed At:** {result['timestamp']}")
            st.write(f"**Confidence:** {prob_percentage:.2f}%")
            
            # Clear button
            if st.button("🔄 Clear Results", use_container_width=True):
                del st.session_state['fraud_result']
                st.rerun()
        else:
            st.info("👈 Fill in transaction details and click **'Analyze Transaction'** to see results")
            
            # Example scenarios
            st.markdown("### 📋 Example Scenarios")
            
            with st.expander("🔴 High Risk Example"):
                st.markdown("""
                - Amount: ₹25,000
                - Time: 3:00 AM (odd hours)
                - Device: New/Unknown
                - Location: Different from usual
                """)
            
            with st.expander("🟢 Low Risk Example"):
                st.markdown("""
                - Amount: ₹500
                - Time: 2:00 PM (business hours)
                - Device: Known device
                - Location: Usual location
                """)

# ==================== TAB 2: CREDIT ELIGIBILITY ====================

with tab2:
    st.header("💰 Credit Eligibility Assessment")
    st.markdown("Banking behavior-based credit eligibility powered by ML")
    
    st.info("ℹ️ **Note:** This model analyzes banking transaction patterns to assess credit eligibility, not traditional loan application data.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Banking Activity Metrics")
        
        with st.form("loan_form"):
            # Account Activity
            st.markdown("**💳 Account Activity**")
            c1, c2 = st.columns(2)
            
            with c1:
                monthly_inflow_avg = st.number_input("Avg Monthly Inflow (₹)", 
                    min_value=0.0, value=50000.0, step=1000.0,
                    help="Average monthly deposits/credits to account")
                avg_transaction_amount = st.number_input("Avg Transaction Amount (₹)", 
                    min_value=0.0, value=2500.0, step=100.0,
                    help="Average amount per transaction")
                max_transaction = st.number_input("Max Transaction (₹)", 
                    min_value=0.0, value=25000.0, step=1000.0,
                    help="Largest single transaction amount")
                active_days = st.slider("Active Days per Month", 
                    min_value=0, max_value=31, value=20,
                    help="Number of days with transactions per month")
            
            with c2:
                transaction_velocity = st.number_input("Transaction Velocity", 
                    min_value=0.0, value=5.5, step=0.1,
                    help="Average transactions per day")
                transaction_amount_volatility = st.slider("Amount Volatility", 
                    min_value=0.0, max_value=1.0, value=0.35, step=0.01,
                    help="Variability in transaction amounts (0=stable, 1=volatile)")
                bounce_rate = st.slider("Bounce Rate (%)", 
                    min_value=0.0, max_value=100.0, value=2.5, step=0.5,
                    help="Percentage of failed/bounced transactions")
                credit_debit_ratio = st.slider("Credit/Debit Ratio", 
                    min_value=0.0, max_value=5.0, value=1.2, step=0.1,
                    help="Ratio of credits to debits (>1 means more credits)")
            
            st.markdown("---")
            
            # Transaction Patterns
            st.markdown("**📈 Transaction Patterns**")
            c3, c4 = st.columns(2)
            
            with c3:
                large_txn_ratio = st.slider("Large Transaction Ratio", 
                    min_value=0.0, max_value=1.0, value=0.15, step=0.01,
                    help="Proportion of transactions >₹10,000")
                small_txn_ratio = st.slider("Small Transaction Ratio", 
                    min_value=0.0, max_value=1.0, value=0.45, step=0.01,
                    help="Proportion of transactions <₹500")
            
            with c4:
                weekend_txn_ratio = st.slider("Weekend Transaction Ratio", 
                    min_value=0.0, max_value=1.0, value=0.28, step=0.01,
                    help="Proportion of transactions on weekends")
                night_txn_ratio = st.slider("Night Transaction Ratio", 
                    min_value=0.0, max_value=1.0, value=0.12, step=0.01,
                    help="Proportion of transactions between 10PM-6AM")
            
            # Auto-calculated insights
            with st.expander("🔍 Calculated Insights", expanded=False):
                st.markdown("**Financial Health Indicators:**")
                col_a, col_b = st.columns(2)
                with col_a:
                    monthly_spend = monthly_inflow_avg / credit_debit_ratio if credit_debit_ratio > 0 else 0
                    st.metric("Est. Monthly Spending", f"₹{monthly_spend:,.0f}")
                    savings_rate = ((credit_debit_ratio - 1) / credit_debit_ratio * 100) if credit_debit_ratio > 0 else 0
                    st.metric("Est. Savings Rate", f"{savings_rate:.1f}%")
                with col_b:
                    txn_per_month = transaction_velocity * active_days
                    st.metric("Transactions/Month", f"{txn_per_month:.0f}")
                    risk_score = (bounce_rate * 2 + transaction_amount_volatility * 50 + night_txn_ratio * 30) / 3
                    st.metric("Behavioral Risk Score", f"{risk_score:.1f}")
            
            submitted_loan = st.form_submit_button("🔍 Check Eligibility", 
                use_container_width=True, type="primary")
            
            if submitted_loan:
                if not DATABRICKS_TOKEN:
                    st.error("❌ Please enter your Databricks token in the sidebar first!")
                else:
                    # Prepare data matching the model's expected schema (12 features)
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
                    
                    with st.spinner("🔍 Analyzing banking behavior with ML model..."):
                        result = predict_loan_eligibility(applicant_data)
                        
                        if result["success"]:
                            data = result["data"]
                            
                            # Store results in session state
                            st.session_state['loan_result'] = {
                                'raw_response': data,
                                'monthly_inflow': monthly_inflow_avg,
                                'bounce_rate': bounce_rate,
                                'credit_debit_ratio': credit_debit_ratio,
                                'transaction_velocity': transaction_velocity,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
                            st.info("💡 Check your token and endpoint status in the sidebar")
                            with st.expander("🔧 Debug Information"):
                                st.json({
                                    "endpoint": LOAN_ENDPOINT,
                                    "payload_features": list(applicant_data.keys()),
                                    "sample_payload": {"dataframe_records": [applicant_data]}
                                })
    
    with col2:
        st.subheader("📊 Eligibility Results")
        
        if 'loan_result' in st.session_state:
            result = st.session_state['loan_result']
            
            st.success("### ✅ Analysis Complete")
            
            # Display raw model output
            st.markdown("#### 🎯 Model Response")
            raw_data = result['raw_response']
            
            if raw_data:
                st.json(raw_data)
                
                st.markdown("---")
                
                # Try to extract common prediction patterns
                if any(key in raw_data for key in ['eligible', 'eligibility', 'approved', 'loan_status']):
                    st.info("✅ Model returned eligibility prediction")
                elif any(key in raw_data for key in ['probability', 'score', 'confidence']):
                    st.info("📊 Model returned probability score")
                else:
                    st.warning("⚠️ Unexpected response format - showing raw output above")
            else:
                st.error("❌ Empty response from model")
            
            st.markdown("---")
            
            # Banking profile summary
            st.markdown("#### 📋 Banking Profile Summary")
            st.write(f"**Avg Monthly Inflow:** ₹{result['monthly_inflow']:,.2f}")
            st.write(f"**Bounce Rate:** {result['bounce_rate']:.1f}%")
            st.write(f"**Credit/Debit Ratio:** {result['credit_debit_ratio']:.2f}")
            st.write(f"**Transaction Velocity:** {result['transaction_velocity']:.1f}/day")
            st.write(f"**Analyzed At:** {result['timestamp']}")
            
            # Clear button
            if st.button("🔄 New Analysis", use_container_width=True):
                del st.session_state['loan_result']
                st.rerun()
            
        else:
            st.info("👈 Fill in banking activity metrics and click **'Check Eligibility'** to see results")
            
            # Guidelines
            st.markdown("### 📋 Healthy Banking Indicators")
            
            st.markdown("""
            **Good Bounce Rate:**
            - <2%: Excellent
            - 2-5%: Good
            - 5-10%: Fair
            - >10%: Poor
            
            **Credit/Debit Ratio:**
            - >1.5: Strong surplus
            - 1.0-1.5: Healthy balance
            - 0.8-1.0: Break-even
            - <0.8: Deficit spending
            
            **Transaction Velocity:**
            - 5-10/day: Active user
            - 2-5/day: Regular user
            - <2/day: Occasional user
            """)

# ==================== TAB 3: DOCUMENTATION ====================

with tab3:
    st.header("📚 API Documentation & Integration Guide")
    
    doc_tab1, doc_tab2, doc_tab3 = st.tabs(["🔌 Endpoints", "💻 Code Examples", "⚙️ Setup"])
    
    with doc_tab1:
        st.markdown("## 🔌 API Endpoints")
        
        st.markdown("### 1. Fraud Detection API")
        st.code(f"{FRAUD_ENDPOINT}", language="text")
        
        st.markdown("**Model:** workspace.fraud_detection.rakshak_fraud_classifier v3")
        st.markdown("**Status:** ✅ READY")
        st.markdown("**Type:** Custom Model (XGBoost + DQN Hybrid)")
        st.markdown("**Features:** 27 transaction features")
        
        st.markdown("---")
        
        st.markdown("### 2. Credit Eligibility API")
        st.code(f"{LOAN_ENDPOINT}", language="text")
        
        st.markdown("**Model:** workspace.fraud_detection.credit_eligibility_classifier v2")
        st.markdown("**Status:** ✅ READY")
        st.markdown("**Type:** Custom Model (Banking Behavior Analysis)")
        st.markdown("**Features:** 12 banking transaction features")
        
        st.markdown("**Required Features:**")
        st.code("""
1. monthly_inflow_avg (double)
2. transaction_velocity (double)
3. bounce_rate (double)
4. avg_transaction_amount (double)
5. transaction_amount_volatility (double)
6. large_txn_ratio (double)
7. max_transaction (double)
8. weekend_txn_ratio (double)
9. night_txn_ratio (double)
10. credit_debit_ratio (double)
11. active_days (double)
12. small_txn_ratio (double)
        """, language="text")
        
        st.markdown("---")
        
        st.markdown("## 🔑 Authentication")
        
        st.markdown("""
        All requests require a Databricks Personal Access Token:
        
        ```
        Authorization: Bearer dapi1234567890abcdef
        Content-Type: application/json
        ```
        
        **Get your token:**
        1. Go to Databricks UI → User Settings
        2. Click "Access Tokens"
        3. Generate New Token
        4. Copy and save securely
        """)
    
    with doc_tab2:
        st.markdown("## 💻 Code Examples")
        
        st.markdown("### Python Example - Fraud Detection")
        
        st.code("""
import requests

url = "https://dbc-a4d8c1e8-b4cf.cloud.databricks.com/serving-endpoints/rakshak-fraud-api/invocations"
token = "your-databricks-token"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

transaction = {
    "day_of_week": 0, "device_enc": 2, "amount_norm": 0.073494,
    "hour_of_day": 3, "txn_velocity": 0.0, "odd_hour_flag": 1,
    "device_network_interaction": 200, "sender_avg_amount_prev": 4303.93,
    "amount_weekend": 0.0, "fuel_large_flag": 0, "amount_vs_sender_mean": 0.853755,
    "high_amount_odd_hour": 0, "cat_device_interaction": 202, "same_bank_flag": 0,
    "sender_age_enc": 1, "weekend_high_spend": 0, "txn_type_enc": 3,
    "high_amount_flag": 0, "amount_hour_interaction": 11.02407,
    "sender_txn_count_prev": 77, "amount_inr": 3674.69, "is_weekend": 0,
    "receiver_age_enc": 0, "network_enc": 0, "sender_max_amount_prev": 6455.895,
    "amount_ratio_deviation": 0.146245, "cat_enc": 2
}

payload = {"dataframe_records": [transaction]}
response = requests.post(url, headers=headers, json=payload, timeout=90)
result = response.json()

print(f"Fraud Probability: {result['predictions'][0]['fraud_probability']:.2%}")
print(f"Is Fraud: {result['predictions'][0]['fraud_prediction'] == 1}")
""", language="python")
        
        st.markdown("---")
        
        st.markdown("### Python Example - Credit Eligibility")
        
        st.code("""
import requests

url = "https://dbc-a4d8c1e8-b4cf.cloud.databricks.com/serving-endpoints/loan-eligibility-api/invocations"
token = "your-databricks-token"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

banking_data = {
    "monthly_inflow_avg": 50000.0,
    "transaction_velocity": 5.5,
    "bounce_rate": 2.5,
    "avg_transaction_amount": 2500.0,
    "transaction_amount_volatility": 0.35,
    "large_txn_ratio": 0.15,
    "max_transaction": 25000.0,
    "weekend_txn_ratio": 0.28,
    "night_txn_ratio": 0.12,
    "credit_debit_ratio": 1.2,
    "active_days": 20.0,
    "small_txn_ratio": 0.45
}

payload = {"dataframe_records": [banking_data]}
response = requests.post(url, headers=headers, json=payload, timeout=90)
result = response.json()

print("Credit Eligibility Result:")
print(result["predictions"][0])
""", language="python")
        
        st.markdown("---")
        
        st.markdown("### cURL Example - Credit Eligibility")
        
        st.code(f"""
curl -X POST "{LOAN_ENDPOINT}" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "dataframe_records": [{{
      "monthly_inflow_avg": 50000.0,
      "transaction_velocity": 5.5,
      "bounce_rate": 2.5,
      "avg_transaction_amount": 2500.0,
      "transaction_amount_volatility": 0.35,
      "large_txn_ratio": 0.15,
      "max_transaction": 25000.0,
      "weekend_txn_ratio": 0.28,
      "night_txn_ratio": 0.12,
      "credit_debit_ratio": 1.2,
      "active_days": 20.0,
      "small_txn_ratio": 0.45
    }}]
  }}'
""", language="bash")
    
    with doc_tab3:
        st.markdown("## ⚙️ Setup & Configuration")
        
        st.markdown("### 1️⃣ Install Dependencies")
        
        st.code("""
pip install streamlit requests pandas python-dotenv
""", language="bash")
        
        st.markdown("### 2️⃣ Create Environment File")
        
        st.markdown("Create a `.env` file in your project directory:")
        
        st.code(f"""
DATABRICKS_HOST={DATABRICKS_HOST}
DATABRICKS_TOKEN=your-token-here
""", language="bash")
        
        st.markdown("### 3️⃣ Run the App")
        
        st.code("""
streamlit run app.py
""", language="bash")
        
        st.markdown("---")
        
        st.markdown("### 🔒 Security Best Practices")
        
        st.markdown("""
        1. **Never commit tokens** to version control
        2. **Use environment variables** for sensitive data
        3. **Rotate tokens regularly** (every 90 days)
        4. **Implement rate limiting** in production
        5. **Use HTTPS only** for API calls
        6. **Add authentication** to your app
        7. **Monitor API usage** for anomalies
        """)
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Information")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Fraud Detection Model**")
            st.markdown("""
            - **Type:** Hybrid (XGBoost + DQN)
            - **Test F1:** 6.05%
            - **Test Recall:** 17.16%
            - **Test Precision:** 3.67%
            - **Optimal Threshold:** 0.100
            - **Features:** 27 inputs
            - **Response:** fraud_probability, fraud_prediction
            """)
        
        with col_b:
            st.markdown("**Credit Eligibility Model**")
            st.markdown("""
            - **Type:** Banking Behavior Classifier
            - **Model Version:** 2
            - **Catalog:** workspace.fraud_detection
            - **Features:** 12 banking metrics
            - **Workload:** CPU
            - **Scale to Zero:** Enabled
            """)
        
        st.markdown("---")
        
        st.markdown("### ⚠️ Important Notes")
        
        st.warning("""
        **Credit Eligibility Model Schema:**
        
        This model analyzes **banking transaction behavior**, not traditional loan application data. 
        It requires 12 specific features related to account activity patterns:
        
        - Monthly inflow averages
        - Transaction velocity and volatility  
        - Bounce rates and ratios
        - Time-based transaction patterns
        
        Do NOT send traditional loan features like age, income, education, etc. 
        The model will reject requests with incorrect schema.
        """)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h4>🛡️ Rakshak Financial Services Platform</h4>
    <p style='color: gray; font-size: 0.9em'>
        Powered by Databricks Model Serving | Built with Streamlit<br>
        ⚡ Real-time ML Inference | 🔒 Enterprise-Grade Security | 📊 Production-Ready
    </p>
    <p style='color: gray; font-size: 0.8em; margin-top: 10px;'>
        Endpoints: rakshak-fraud-api (v3) • loan-eligibility-api (v2)
    </p>
</div>
""", unsafe_allow_html=True)
