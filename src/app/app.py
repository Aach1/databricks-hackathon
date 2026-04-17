import os
import streamlit as st
import pandas as pd
import numpy as np
from databricks.sdk.core import Config
from databricks import sql

# 1. Configuration & Auth
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

@st.cache_resource
def get_db_connection():
    # Use Databricks SDK Config to get credentials automatically
    cfg = Config()
    
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        st.warning("DATABRICKS_WAREHOUSE_ID not set. Using mock data.")
        return None
        
    try:
        conn = sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            credentials_provider=lambda: cfg.authenticate
        )
        return conn
    except Exception as e:
        st.error(f"Failed to connect to Databricks: {e}")
        return None

def fetch_data(conn, limit=100):
    catalog = os.getenv("DATABRICKS_CATALOG", "main")
    schema = os.getenv("DATABRICKS_SCHEMA", "fraud_detection_dev")
    table = "transactions"
    
    query = f"""
    SELECT transaction_id, user_id, amount, category, timestamp, is_fraud 
    FROM {catalog}.{schema}.{table}
    ORDER BY timestamp DESC
    LIMIT {limit}
    """
    
    if conn is None:
        # Return mock data
        return pd.DataFrame({
            'transaction_id': [f"TXN-{i}" for i in range(limit)],
            'user_id': np.random.randint(1000, 9999, limit),
            'amount': np.round(np.random.uniform(10, 1500, limit), 2),
            'category': np.random.choice(['Retail', 'Travel', 'Dining', 'Online Shopping'], limit),
            'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=limit),
            'is_fraud': np.random.choice([0, 1], limit, p=[0.95, 0.05])
        })
        
    with conn.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)

# UI Implementation
st.title("🛡️ Hybrid Fraud Detection System")
st.markdown("Monitor real-time transactions scored by the **XGBoost + DQN Agent**.")

conn = get_db_connection()
df = fetch_data(conn, limit=50)

# Dashboard Metrics
st.subheader("System Overview")
col1, col2, col3, col4 = st.columns(4)
fraud_count = df['is_fraud'].sum()
col1.metric("Recent Transactions", len(df))
col2.metric("Flagged as Fraud", fraud_count, f"{(fraud_count/len(df))*100:.1f}%")
col3.metric("Agent Status", "Active")
col4.metric("Avg Review Time", "1.2s")

st.divider()

# Main Data Table
st.subheader("Recent Transactions")

# Highlight fraudulent rows
def color_fraud(val):
    color = '#ff4b4b' if val == 1 else ''
    return f'background-color: {color}'

st.dataframe(
    df.style.map(color_fraud, subset=['is_fraud']),
    use_container_width=True,
    hide_index=True
)

st.sidebar.header("DQN Agent Controls")
st.sidebar.markdown(
    """
    **Current Policy:** Balance
    - False Positive Penalty: High
    - Fraud Loss Penalty: Critical
    """
)
if st.sidebar.button("Retrain Agent (Simulated)"):
    st.sidebar.success("Training job triggered via Databricks Jobs!")
