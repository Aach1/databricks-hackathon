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
 page_title="Rakshak - Financial Intelligence Platform",
 page_icon="",
 layout="wide",
 initial_sidebar_state="expanded"
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
     model_uri = "runs:/81848d9871fc49c7987e233bced963eb/model"
     model = mlflow.pyfunc.load_model(model_uri)
     return model
 except Exception as e:
     st.error(f"Failed to load loan model: {str(e)}")
     return None

@st.cache_resource
def load_rag_models():
 """Load RAG pipeline models (Sarvam-1 + Embeddings)"""
 try:
 try:
     from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
     from sentence_transformers import SentenceTransformer
 except ImportError as e:
     return {
     "success": False, 
     "error": f"Missing dependencies: {str(e)}. Please restart the app to install requirements.txt"
     }
     import faiss
 
     st.info(" Loading Sarvam-1 LLM (8-bit quantized, ~3.5GB)... This may take 5-10 minutes on first load.")
 
     # Configure 8-bit quantization for Sarvam-1
     quantization_config = BitsAndBytesConfig(
     load_in_8bit=True,
     llm_int8_threshold=6.0
     )
 
     model_name = "sarvamai/sarvam-1"
 
     # Load tokenizer
     tokenizer = AutoTokenizer.from_pretrained(model_name)
 
     # Load model with 8-bit quantization
     model = AutoModelForCausalLM.from_pretrained(
     model_name,
     quantization_config=quantization_config,
     device_map="auto",
     low_cpu_mem_usage=True
     )
 
     # Load multilingual embedding model
     st.info(" Loading multilingual embedding model (~470MB)...")
     embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
 
     return {
     "llm_model": model,
     "llm_tokenizer": tokenizer,
     "embedding_model": embedding_model,
     "success": True
     }
 except Exception as e:
     st.error(f"Failed to load RAG models: {str(e)}")
     return {"success": False, "error": str(e)}

     # ==================== HELPER FUNCTIONS ====================

def predict_fraud(transaction_data, model):
 """Predict fraud using loaded model"""
 try:
     # Convert to DataFrame
     df = pd.DataFrame([transaction_data])
 
     # Cast integer columns to int32 to match model schema
     int_cols = ['day_of_week', 'device_enc', 'hour_of_day', 'odd_hour_flag', 
     'device_network_interaction', 'fuel_large_flag', 'high_amount_odd_hour',
     'cat_device_interaction', 'same_bank_flag', 'sender_age_enc', 
     'weekend_high_spend', 'txn_type_enc', 'high_amount_flag', 
     'is_weekend', 'receiver_age_enc', 'network_enc', 'cat_enc']
     for col in int_cols:
     df[col] = df[col].astype('int32')
 
     # Cast long column
     df['sender_txn_count_prev'] = df['sender_txn_count_prev'].astype('int64')
 
     # Cast double columns to float64
     double_cols = ['amount_norm', 'txn_velocity', 'sender_avg_amount_prev', 
     'amount_weekend', 'amount_vs_sender_mean', 'amount_hour_interaction',
     'amount_inr', 'sender_max_amount_prev', 'amount_ratio_deviation']
     for col in double_cols:
     df[col] = df[col].astype('float64')
 
     # Predict
     prediction = model.predict(df)
 
     # Parse prediction
     if isinstance(prediction, pd.DataFrame):
     fraud_prob = float(prediction["fraud_probability"].iloc[0])
     fraud_pred = int(prediction["fraud_prediction"].iloc[0])
     elif isinstance(prediction, np.ndarray):
     fraud_prob = float(prediction[0][0] if len(prediction[0]) > 0 else prediction[0])
     fraud_pred = int(fraud_prob > 0.1)
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
 return "", "#ff4444"
 elif probability >= 0.4:
 return "", "#ffaa00"
 else:
 return "", "#44ff44"

# ==================== RAG PIPELINE FUNCTIONS ====================

def generate_text_rag(prompt, model, tokenizer, max_tokens=100, temperature=0.7):
 """Generate text using Sarvam-1"""
 try:
     inputs = tokenizer(prompt, return_tensors="pt")
 
     with torch.no_grad():
     outputs = model.generate(
     **inputs,
     max_new_tokens=max_tokens,
     do_sample=True,
     temperature=temperature,
     pad_token_id=tokenizer.eos_token_id
     )
 
     generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
     result = generated.replace(prompt, "").strip()
     return result
 except Exception as e:
     return f"Error generating text: {str(e)}"

def detect_language(text):
 """Simple language detection based on script"""
 # Check for different scripts
 if any('\u0900' <= c <= '\u097F' for c in text): # Devanagari (Hindi, Marathi)
 if 'मराठी' in text or 'आहे' in text:
 return "Marathi"
 return "Hindi"
 elif any('\u0B80' <= c <= '\u0BFF' for c in text): # Tamil
 return "Tamil"
 elif any('\u0C00' <= c <= '\u0C7F' for c in text): # Telugu
 return "Telugu"
 elif any('\u0980' <= c <= '\u09FF' for c in text): # Bengali
 return "Bengali"
 elif any('\u0A80' <= c <= '\u0AFF' for c in text): # Gujarati
 return "Gujarati"
 elif any('\u0C80' <= c <= '\u0CFF' for c in text): # Kannada
 return "Kannada"
 elif any('\u0D00' <= c <= '\u0D7F' for c in text): # Malayalam
 return "Malayalam"
 elif any('\u0A00' <= c <= '\u0A7F' for c in text): # Punjabi
 return "Punjabi"
 else:
 return "English"

def simple_rag_query(question, knowledge_base, embedding_model, llm_model, llm_tokenizer, top_k=2, selected_language="English"):
 """Simple RAG query with language-aware response"""
 try:
     # Detect question language
     detected_lang = detect_language(question)
 
     # Embed question
     question_emb = embedding_model.encode([question])
 
     # Embed knowledge base
     kb_texts = [doc["text"] for doc in knowledge_base]
     kb_embs = embedding_model.encode(kb_texts)
 
     # Calculate similarities
     from numpy import dot
     from numpy.linalg import norm
 
     similarities = []
     for kb_emb in kb_embs:
     sim = dot(question_emb[0], kb_emb) / (norm(question_emb[0]) * norm(kb_emb))
     similarities.append(sim)
 
     # Get top-k indices
     top_indices = np.argsort(similarities)[-top_k:][::-1]
 
     # Prioritize same-language documents if available
     same_lang_docs = []
     other_lang_docs = []
 
     for idx in top_indices:
     doc = knowledge_base[idx]
     if doc.get("language", "English") == detected_lang:
     same_lang_docs.append(doc["text"])
     else:
     other_lang_docs.append(doc["text"])
 
     # Combine: same language first, then others
     context_parts = same_lang_docs + other_lang_docs
     context = "\n\n".join(context_parts[:top_k])
 
     # Language-specific instruction templates
     lang_instructions = {
     "English": "Answer in English only.",
     "Hindi": "केवल हिंदी में उत्तर दें।",
     "Tamil": "தமிழில் மட்டும் பதிலளிக்கவும்.",
     "Telugu": "తెలుగులో మాత్రమే సమాధానం ఇవ్వండి.",
     "Bengali": "শুধুমাত্র বাংলায় উত্তর দিন।",
     "Marathi": "फक्त मराठीत उत्तर द्या।",
     "Gujarati": "ફક્ત ગુજરાતીમાં જવાબ આપો.",
     "Kannada": "ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸಿ.",
     "Malayalam": "മലയാളത്തിൽ മാത്രം ഉത്തരം നൽകുക.",
     "Punjabi": "ਕੇਵਲ ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦਿਓ।"
     }
 
     # Get instruction in detected language
     lang_instruction = lang_instructions.get(detected_lang, f"Answer only in {detected_lang}.")
 
     # Create prompt with explicit language instruction
     prompt = f"""Context:
     {context}

     Question: {question}

     Important: {lang_instruction}
     Answer the question based on the context above."""
 
     # Generate
     answer = generate_text_rag(prompt, llm_model, llm_tokenizer, max_tokens=100, temperature=0.7)
 
     return {
     "success": True,
     "answer": answer,
     "context": context,
     "detected_language": detected_lang
     }
 except Exception as e:
     return {"success": False, "error": str(e)}

     # ==================== MAIN APP ====================

     st.title("Rakshak - Financial Intelligence Platform")
     st.markdown("Real-time ML-powered Fraud Detection | Credit Analysis | Multilingual AI Assistant")
     st.markdown("---")

     # Load models
     with st.spinner("Loading ML models..."):
     fraud_model = load_fraud_model()
     loan_model = load_loan_model()

     # Sidebar for model status
     with st.sidebar:
     st.header("System Configuration")
    
     st.subheader("Model Status")
    
     if fraud_model:
        st.success("Fraud Detection: Loaded")
     else:
        st.error("Fraud Detection: Failed")
    
     if loan_model:
        st.success("Credit Eligibility: Loaded")
     else:
        st.error("Credit Eligibility: Failed")
    
     st.markdown("---")
    
     st.subheader("System Info")
     st.text("Inference: In-App")
     st.text("Models: 3 Loaded")
     st.text("Languages: 10+")
     st.text("Status: Production")
 
     st.markdown("<h3 style='color: white;'>[STATUS] Model Status</h3>", unsafe_allow_html=True)
 
     col1, col2 = st.columns([1, 4])
     with col1:
     if fraud_model:
     st.markdown("[OK]", unsafe_allow_html=True)
     else:
     st.markdown("[X]", unsafe_allow_html=True)
     with col2:
     st.markdown("<p style='color: white; margin: 0;'>Fraud Detection</p>", unsafe_allow_html=True)
 
     col1, col2 = st.columns([1, 4])
     with col1:
     if loan_model:
     st.markdown("[OK]", unsafe_allow_html=True)
     else:
     st.markdown("[X]", unsafe_allow_html=True)
     with col2:
     st.markdown("<p style='color: white; margin: 0;'>Credit Eligibility</p>", unsafe_allow_html=True)
 
     st.markdown("---")
     st.markdown("**Rakshak Financial Intelligence Platform**")
     st.caption("Fraud Detection | Credit Eligibility | Multilingual AI Assistant | Powered by Sarvam-1 LLM")
 
     st.markdown("<h3 style='color: white;'>[INFO] System Info</h3>", unsafe_allow_html=True)
 
     info_items = [
     ("", "Inference", "In-App (No endpoints)"),
     ("", "Models", "3 AI Models Loaded"),
     ("", "Status", "Production Ready"),
     ("[AI] ", "Languages", "10+ Supported")
     ]
 
     for icon, label, value in info_items:
     st.markdown(f"""
     <div style='background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem;'>
     <p style='color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;'>{icon} {label}</p>
     <p style='color: white; margin: 0; font-weight: 600;'>{value}</p>
     </div>
     """, unsafe_allow_html=True)

     # ==================== TABS ====================

     tab1, tab2, tab3, tab4 = st.tabs([
     "[FRAUD] Fraud Detection", 
     "[CREDIT] Credit Eligibility", 
     "[AI] Multilingual AI Assistant (RAG)",
     "[DOCS] Documentation"
     ])

     # ==================== TAB 1: FRAUD DETECTION ====================

     with tab1:
     st.header("[FRAUD] UPI Transaction Fraud Detection")
     st.markdown("Analyze real-time UPI transactions for potential fraud using our hybrid XGBoost + DQN model.")
 
     if not fraud_model:
     st.error("[X] Fraud detection model not loaded. Cannot perform predictions.")
     else:
     col1, col2 = st.columns([2, 1])
 
     with col1:
     st.subheader("Transaction Details")
 
     with st.form("fraud_form"):
     c1, c2, c3 = st.columns(3)
 
     with c1:
     st.markdown("**[CREDIT] Transaction Info**")
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
     st.markdown("** Time Details**")
     hour_of_day = st.slider("Hour of Day", 0, 23, 3)
     day_of_week = st.slider("Day of Week", 0, 6, 0)
     is_weekend = st.checkbox("Is Weekend", value=False)
     network_enc = st.selectbox("Network Type", 
     options=[0, 1, 2], 
     format_func=lambda x: ["WiFi", "4G", "5G"][x],
     index=0)
 
     with c3:
     st.markdown("** User Details**")
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
 
     submitted = st.form_submit_button(" Analyze Transaction", use_container_width=True, type="primary")
 
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
 
     with st.spinner(" Analyzing transaction..."):
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
     st.error(f"[X] Error: {result['error']}")
 
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
     st.error("### [FRAUD] FRAUD DETECTED!")
     st.markdown("""
     **Recommended Actions:**
     - Block transaction immediately
     - Contact sender for verification
     - [FRAUD] Flag account for investigation
     """)
     else:
     st.success("### [OK] Transaction Appears Safe")
     st.markdown("""
     **Next Steps:**
     - [OK] Process transaction normally
     - [STATUS] Continue monitoring
     """)
 
     st.markdown("---")
     st.markdown("#### Transaction Summary")
     st.write(f"**Amount:** ₹{result['amount']:,.2f}")
     st.write(f"**Analyzed At:** {result['timestamp']}")
 
     if st.button(" Clear Results", use_container_width=True):
     del st.session_state['fraud_result']
     st.rerun()
     else:
     st.info("👈 Fill in transaction details to see results")

     # ==================== TAB 2: CREDIT ELIGIBILITY ====================

     with tab2:
     st.header("[CREDIT] Credit Eligibility Assessment")
     st.markdown("Banking behavior-based credit eligibility powered by ML")
 
     if not loan_model:
     st.error("[X] Credit eligibility model not loaded. Cannot perform predictions.")
     else:
     col1, col2 = st.columns([2, 1])
 
     with col1:
     st.subheader("[STATUS] Banking Activity Metrics")
 
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
 
     st.markdown("** Transaction Patterns**")
     c3, c4 = st.columns(2)
 
     with c3:
     large_txn_ratio = st.slider("Large Transaction Ratio", 0.0, 1.0, 0.15, 0.01)
     small_txn_ratio = st.slider("Small Transaction Ratio", 0.0, 1.0, 0.45, 0.01)
 
     with c4:
     weekend_txn_ratio = st.slider("Weekend Transaction Ratio", 0.0, 1.0, 0.28, 0.01)
     night_txn_ratio = st.slider("Night Transaction Ratio", 0.0, 1.0, 0.12, 0.01)
 
     submitted_loan = st.form_submit_button(" Check Eligibility", 
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
 
     with st.spinner(" Analyzing banking behavior..."):
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
     st.error(f"[X] Error: {result['error']}")
 
     with col2:
     st.subheader("[STATUS] Eligibility Results")
 
     if 'loan_result' in st.session_state:
     result = st.session_state['loan_result']
 
     prediction = result['raw_response'].get('prediction', -1)
     is_eligible = prediction == 1
 
     if is_eligible:
     st.success("### [OK] CREDIT ELIGIBLE")
     st.markdown("""**Decision:** Approved for credit services
 
     **Banking behavior indicates:**
     - [OK] Stable transaction patterns
     - [OK] Acceptable bounce rate
     - [OK] Sufficient monthly inflow
     """)
     else:
     st.error("### [X] NOT ELIGIBLE")
     st.markdown("""**Decision:** Credit application declined
 
     **Reasons may include:**
     - [!] High bounce rate
     - [!] Irregular transaction patterns
     - [!] Insufficient banking activity
     """)
 
     st.markdown("---")
     st.markdown("#### [STATUS] Prediction Details")
     st.write(f"**Model Output:** {prediction} ({'Eligible' if is_eligible else 'Not Eligible'})")
     st.write(f"**Confidence:** Based on 12 banking metrics")
 
     st.markdown("---")
     st.markdown("#### Banking Profile")
     st.write(f"**Avg Monthly Inflow:** ₹{result['monthly_inflow']:,.2f}")
     st.write(f"**Bounce Rate:** {result['bounce_rate']:.1f}%")
     st.write(f"**Analyzed At:** {result['timestamp']}")
 
     if st.button(" New Analysis", use_container_width=True):
     del st.session_state['loan_result']
     st.rerun()
     else:
     st.info("👈 Fill in banking metrics to see results")

     # ==================== TAB 3: MULTILINGUAL AI ASSISTANT (RAG) ====================

     with tab3:
     st.header("[AI] Multilingual AI Assistant - Sarvam-1 RAG")
     st.markdown("**Retrieval-Augmented Generation for Questions in Any Language**")
     st.markdown("Ask questions in Hindi, Tamil, Telugu, Bengali, English, and 50+ languages!")
 
     # Load RAG models (cached)
     if 'rag_models' not in st.session_state:
     with st.spinner(" Loading Sarvam-1 LLM and embedding models... This may take 5-10 minutes..."):
     st.session_state['rag_models'] = load_rag_models()
 
     rag_models = st.session_state['rag_models']
 
     if not rag_models.get("success", False):
     st.error(f"[X] RAG models failed to load: {rag_models.get('error', 'Unknown error')}")
     st.info(" RAG requires Sarvam-1 LLM and multilingual embeddings. Please ensure compute has sufficient memory (8GB+).")
     else:
     st.success("[OK] Sarvam-1 LLM and embeddings loaded successfully!")
 
     # Multilingual knowledge base
     if 'knowledge_base' not in st.session_state:
     st.session_state['knowledge_base'] = [
     # Hindi
     {
     "text": "भारत में UPI (यूनिफाइड पेमेंट्स इंटरफेस) एक डिजिटल भुगतान प्रणाली है। यह NPCI द्वारा विकसित की गई है। UPI से आप तुरंत पैसे ट्रांसफर कर सकते हैं। PhonePe, Google Pay, और Paytm लोकप्रिय UPI ऐप हैं।",
     "language": "Hindi",
     "topic": "upi"
     },
     # English
     {
     "text": "UPI (Unified Payments Interface) is India's instant payment system developed by NPCI. It allows instant money transfer between bank accounts using mobile phones. Popular UPI apps include PhonePe, Google Pay, and Paytm. UPI transactions are free for individuals.",
     "language": "English",
     "topic": "upi"
     },
     # Tamil
     {
     "text": "UPI (யூனிஃபைட் பேமெண்ட்ஸ் இன்டர்ஃபேஸ்) என்பது இந்தியாவின் உடனடி பணம் பரிமாற்ற முறையாகும். NPCI-யால் உருவாக்கப்பட்டது. PhonePe, Google Pay, Paytm போன்றவை பிரபலமான UPI பயன்பாடுகள்.",
     "language": "Tamil",
     "topic": "upi"
     },
     # Hindi - Fraud
     {
     "text": "धोखाधड़ी का पता लगाने के लिए मशीन लर्निंग का उपयोग किया जाता है। संदिग्ध लेनदेन को पहचानने के लिए XGBoost और न्यूरल नेटवर्क जैसे एल्गोरिदम उपयोग होते हैं। उच्च राशि, असामान्य समय, और असामान्य स्थान संकेतक हैं।",
     "language": "Hindi",
     "topic": "fraud"
     },
     # English - Fraud
     {
     "text": "Machine learning is used for fraud detection in financial transactions. Algorithms like XGBoost and Neural Networks identify suspicious patterns. High transaction amounts, unusual timing, abnormal locations, and irregular behavior are key fraud indicators.",
     "language": "English",
     "topic": "fraud"
     },
     # Telugu
     {
     "text": "మోసాన్ని గుర్తించడానికి మెషిన్ లెర్నింగ్ ఉపయోగించబడుతుంది. XGBoost మరియు న్యూరల్ నెట్‌వర్క్‌లు అనుమానాస్పద లావాదేవీలను గుర్తిస్తాయి. అధిక మొత్తం, అసాధారణ సమయం సంకేతాలు.",
     "language": "Telugu",
     "topic": "fraud"
     },
     # Hindi - Loan
     {
     "text": "ऋण पात्रता (Loan Eligibility) आपके बैंकिंग व्यवहार पर निर्भर करती है। मासिक आय, लेनदेन पैटर्न, बाउंस दर, और क्रेडिट इतिहास महत्वपूर्ण कारक हैं। अच्छी बैंकिंग आदतें आपकी पात्रता बढ़ाती हैं।",
     "language": "Hindi",
     "topic": "loan"
     },
     # English - Loan
     {
     "text": "Loan eligibility depends on your banking behavior. Key factors include monthly income, transaction patterns, bounce rate, and credit history. Good banking habits improve eligibility. Regular transactions and low bounce rates are positive indicators.",
     "language": "English",
     "topic": "loan"
     },
     # Bengali
     {
     "text": "ঋণের যোগ্যতা আপনার ব্যাংকিং আচরণের উপর নির্ভর করে। মাসিক আয়, লেনদেন প্যাটার্ন, বাউন্স হার এবং ক্রেডিট ইতিহাস গুরুত্বপূর্ণ কারণ। ভাল ব্যাংকিং অভ্যাস যোগ্যতা বৃদ্ধি করে।",
     "language": "Bengali",
     "topic": "loan"
     },
     # Hindi - Sarvam
     {
     "text": "Sarvam-1 एक भारतीय भाषा मॉडल है जो हिंदी और अन्य भारतीय भाषाओं को समझता है। यह 7 बिलियन पैरामीटर वाला मॉडल है। Sarvam AI द्वारा विकसित किया गया है। यह तमिल, तेलुगु, बंगाली, मराठी जैसी भाषाओं को support करता है।",
     "language": "Hindi",
     "topic": "sarvam"
     },
     # English - Sarvam
     {
     "text": "Sarvam-1 is an Indian language model that understands Hindi and other Indian languages. It's a 7 billion parameter model developed by Sarvam AI. Supports Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, and more.",
     "language": "English",
     "topic": "sarvam"
     }
     ]
 
     # Language selector
     st.markdown("---")
     col_lang, col_info = st.columns([1, 2])
 
     with col_lang:
     selected_language = st.selectbox(
     " Select Language",
     ["English", "Hindi (हिंदी)", "Tamil (தமிழ்)", "Telugu (తెలుగు)", 
     "Bengali (বাংলা)", "Marathi (मराठी)", "Gujarati (ગુજરાતી)", 
     "Kannada (ಕನ್ನಡ)", "Malayalam (മലയാളം)", "Punjabi (ਪੰਜਾਬੀ)"],
     index=0
     )
 
     with col_info:
     st.info(" Ask questions in any language! The model will respond in the same language.")
 
     col1, col2 = st.columns([2, 1])
 
     with col1:
     st.subheader(" Ask a Question")
 
     # Sample questions by language
     sample_questions_by_lang = {
     "English": [
     "What is UPI?",
     "How is fraud detected?",
     "What factors affect loan eligibility?",
     "What is Sarvam-1?"
     ],
     "Hindi (हिंदी)": [
     "UPI क्या है?",
     "धोखाधड़ी का पता कैसे लगाया जाता है?",
     "ऋण पात्रता के लिए क्या जरूरी है?",
     "Sarvam-1 क्या है?"
     ],
     "Tamil (தமிழ்)": [
     "UPI என்றால் என்ன?",
     "மோசடி எப்படி கண்டறியப்படுகிறது?",
     "கடன் தகுதிக்கு என்ன தேவை?",
     "Sarvam-1 என்றால் என்ன?"
     ],
     "Telugu (తెలుగు)": [
     "UPI అంటే ఏమిటి?",
     "మోసం ఎలా గుర్తించబడుతుంది?",
     "రుణ అర్హతకు ఏమి అవసరం?",
     "Sarvam-1 అంటే ఏమిటి?"
     ],
     "Bengali (বাংলা)": [
     "UPI কী?",
     "প্রতারণা কীভাবে সনাক্ত করা হয়?",
     "ঋণের যোগ্যতার জন্য কী প্রয়োজন?",
     "Sarvam-1 কী?"
     ],
     "Marathi (मराठी)": [
     "UPI म्हणजे काय?",
     "फसवणूक कशी शोधली जाते?",
     "कर्जाच्या पात्रतेसाठी काय आवश्यक आहे?",
     "Sarvam-1 म्हणजे काय?"
     ],
     "Gujarati (ગુજરાતી)": [
     "UPI શું છે?",
     "છેતરપિંડી કેવી રીતે શોધવામાં આવે છે?",
     "લોન પાત્રતા માટે શું જરૂરી છે?",
     "Sarvam-1 શું છે?"
     ],
     "Kannada (ಕನ್ನಡ)": [
     "UPI ಎಂದರೇನು?",
     "ವಂಚನೆಯನ್ನು ಹೇಗೆ ಪತ್ತೆ ಮಾಡಲಾಗುತ್ತದೆ?",
     "ಸಾಲದ ಅರ್ಹತೆಗೆ ಏನು ಅಗತ್ಯ?",
     "Sarvam-1 ಎಂದರೇನು?"
     ],
     "Malayalam (മലയാളം)": [
     "UPI എന്താണ്?",
     "തട്ടിപ്പ് എങ്ങനെ കണ്ടെത്തുന്നു?",
     "വായ്പാ യോഗ്യതയ്ക്ക് എന്താണ് ആവശ്യം?",
     "Sarvam-1 എന്താണ്?"
     ],
     "Punjabi (ਪੰਜਾਬੀ)": [
     "UPI ਕੀ ਹੈ?",
     "ਧੋਖਾਧੜੀ ਦਾ ਪਤਾ ਕਿਵੇਂ ਲਗਾਇਆ ਜਾਂਦਾ ਹੈ?",
     "ਕਰਜ਼ੇ ਦੀ ਯੋਗਤਾ ਲਈ ਕੀ ਜ਼ਰੂਰੀ ਹੈ?",
     "Sarvam-1 ਕੀ ਹੈ?"
     ]
     }
 
     # Get samples for selected language
     lang_key = selected_language
     samples = sample_questions_by_lang.get(lang_key, sample_questions_by_lang["English"])
 
     st.markdown(f"** Sample Questions in {selected_language.split()[0]}:**")
     selected_sample = st.radio("Or choose from below:", samples, index=0)
 
     # Custom question input
     user_question = st.text_area(
     f"Write your question in {selected_language.split()[0]}:",
     value=selected_sample,
     height=100
     )
 
     # Parameters
     col_a, col_b = st.columns(2)
     with col_a:
     max_tokens = st.slider("Max Tokens (Answer Length)", 50, 200, 100)
     with col_b:
     top_k = st.slider("Top-K Documents", 1, 4, 2)
 
     if st.button(" Ask Question", type="primary", use_container_width=True):
     if user_question.strip():
     with st.spinner(" Thinking... (This may take 2-5 minutes on CPU)"):
     result = simple_rag_query(
     user_question,
     st.session_state['knowledge_base'],
     rag_models['embedding_model'],
     rag_models['llm_model'],
     rag_models['llm_tokenizer'],
     top_k=top_k,
     selected_language=selected_language
     )
 
     if result["success"]:
     st.session_state['rag_result'] = {
     'question': user_question,
     'answer': result['answer'],
     'context': result['context'],
     'language': selected_language,
     'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
     }
     st.rerun()
     else:
     st.error(f"[X] Error: {result['error']}")
     else:
     st.warning("[!] Please enter a question")
 
     with col2:
     st.subheader("[STATUS] Answer")
 
     if 'rag_result' in st.session_state:
     result = st.session_state['rag_result']
 
     st.markdown("### Question")
     st.info(result['question'])
 
     st.markdown("### Answer")
     st.success(result['answer'])
 
     with st.expander("[DOCS] Retrieved Context"):
     st.text(result['context'])
 
     st.markdown("---")
     st.caption(f" Language: {result.get('language', 'Auto-detected')}")
     st.caption(f" Answered at: {result['timestamp']}")
 
     if st.button(" Clear Answer", use_container_width=True):
     del st.session_state['rag_result']
     st.rerun()
     else:
     st.info("👈 Ask a question to see the answer here")
 
     st.markdown("---")
     st.markdown("#### [DOCS] Current Knowledge Base")
 
     # Count documents by language
     lang_counts = {}
     for doc in st.session_state['knowledge_base']:
     lang = doc.get('language', 'Unknown')
     lang_counts[lang] = lang_counts.get(lang, 0) + 1
 
     lang_summary = ", ".join([f"{lang} ({count})" for lang, count in lang_counts.items()])
     st.markdown(f"**{len(st.session_state['knowledge_base'])} documents** in {len(lang_counts)} languages: {lang_summary}")
     st.markdown("**Topics:** UPI, Fraud Detection, Loan Eligibility, Sarvam-1")
 
     with st.expander(" Add Custom Document (Any Language)"):
     custom_doc = st.text_area("Add text to knowledge base (any language):", height=100)
     custom_topic = st.text_input("Topic:", "custom")
     custom_lang = st.text_input("Language:", "English")
     if st.button("Add to Knowledge Base"):
     if custom_doc.strip():
     st.session_state['knowledge_base'].append({
     "text": custom_doc,
     "topic": custom_topic,
     "language": custom_lang
     })
     st.success(f"[OK] Document added in {custom_lang}!")
     st.rerun()

     # ==================== TAB 4: DOCUMENTATION ====================

     with tab4:
     st.header("[DOCS] Model Documentation")
 
     st.markdown("## 🎯 Inference Method")
     st.info("[OK] **In-App Inference:** Models loaded directly from MLflow - No Model Serving endpoints used")
 
     st.markdown("### 1. Fraud Detection Model")
     st.code("""
     Model URI: runs:/cabe6e4aac7f42bc80e46f3fa402e885/hybrid_fraud_model
     Type: Hybrid (XGBoost + DQN)
     Features: 27 transaction features
     Output: fraud_probability, fraud_prediction
     """)
 
     st.markdown("### 2. Credit Eligibility Model")
     st.code("""
     Model URI: runs:/81848d9871fc49c7987e233bced963eb/model
     Type: Banking Behavior Classifier
     Features: 12 banking metrics
     Output: eligibility score/prediction
     """)
 
     st.markdown("### 3. Sarvam-1 Multilingual RAG Pipeline (NEW)")
     st.code("""
     LLM: sarvamai/sarvam-1 (7B parameters, 8-bit quantized)
     Memory: ~3.5GB (CPU-optimized)
     Embeddings: paraphrase-multilingual-MiniLM-L12-v2 (~470MB)
     Languages Supported: 50+ languages
     - Indian: Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, 
     Kannada, Malayalam, Punjabi, Odia
     - International: English, Spanish, French, German, Chinese, and more
     Retrieval: Semantic similarity search
     Generation: Context-aware Hindi text generation
     """)
 
     st.markdown("### [STATUS] Total Memory Footprint")
     st.info("""
     - Fraud Model: ~50MB
     - Loan Model: ~10MB 
     - Sarvam-1 LLM: ~3.5GB (8-bit quantized)
     - Embedding Model: ~470MB
     - **Total: ~4GB** (fits within CPU memory constraints)
     """)

     # ==================== FOOTER ====================

     st.markdown("---")
     st.markdown("**Rakshak Financial Intelligence Platform**")
     st.caption("Fraud Detection | Credit Eligibility | Multilingual AI Assistant | Powered by Sarvam-1 LLM")
     st.markdown("""
     <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%); border-radius: 15px;'>
     <h2 style='margin: 0 0 1rem 0;'> Rakshak Financial Intelligence</h2>
     <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-bottom: 1rem;'>
     <div>
     <p style='color: #667eea; font-weight: 700; margin: 0; font-size: 1.5rem;'>3</p>
     <p style='color: #666; margin: 0; font-size: 0.9rem;'>AI Models</p>
     </div>
     <div>
     <p style='color: #764ba2; font-weight: 700; margin: 0; font-size: 1.5rem;'>10+</p>
     <p style='color: #666; margin: 0; font-size: 0.9rem;'>Languages</p>
     </div>
     <div>
     <p style='color: #667eea; font-weight: 700; margin: 0; font-size: 1.5rem;'>100%</p>
     <p style='color: #666; margin: 0; font-size: 0.9rem;'>CPU Inference</p>
     </div>
     <div>
     <p style='color: #764ba2; font-weight: 700; margin: 0; font-size: 1.5rem;'>0</p>
     <p style='color: #666; margin: 0; font-size: 0.9rem;'>Endpoints</p>
     </div>
     </div>
     <p style='color: #666; font-size: 0.9em; margin: 0.5rem 0;'>
     [FRAUD] Fraud Detection | [CREDIT] Credit Analysis | [AI] Multilingual AI Assistant
     </p>
     <p style='color: #999; font-size: 0.8em; margin: 0;'>
     Powered by Sarvam-1 LLM | In-App ML Inference | Hackathon Compliant
     </p>
     </div>
     """, unsafe_allow_html=True)
