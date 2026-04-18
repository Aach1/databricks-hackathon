import streamlit as st
import mlflow
import pandas as pd
import numpy as np
import torch

st.set_page_config(page_title="Rakshak", page_icon=":shield:", layout="wide")
mlflow.set_tracking_uri("databricks")

@st.cache_resource
def load_fraud_model():
    try:
        return mlflow.pyfunc.load_model("runs:/cabe6e4aac7f42bc80e46f3fa402e885/hybrid_fraud_model")
    except: return None

@st.cache_resource
def load_loan_model():
    try:
        return mlflow.pyfunc.load_model("runs:/81848d9871fc49c7987e233bced963eb/model")
    except: return None

@st.cache_resource
def load_rag_models():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from sentence_transformers import SentenceTransformer
        st.info("Loading Sarvam-1 (8-bit, ~3.5GB)... This takes 5-10 minutes first time.")
        config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")
        model = AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-1", quantization_config=config, device_map="auto", low_cpu_mem_usage=True)
        st.info("Loading embeddings (~470MB)...")
        embed = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return {"llm_model": model, "llm_tokenizer": tokenizer, "embedding_model": embed, "success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

def predict_fraud(txn, model):
    try:
        df = pd.DataFrame([txn])
        int_cols = ['day_of_week', 'device_enc', 'hour_of_day', 'odd_hour_flag', 'device_network_interaction', 
                    'fuel_large_flag', 'high_amount_odd_hour', 'cat_device_interaction', 'same_bank_flag', 
                    'sender_age_enc', 'weekend_high_spend', 'txn_type_enc', 'high_amount_flag', 'is_weekend', 
                    'receiver_age_enc', 'network_enc', 'cat_enc']
        for col in int_cols:
            df[col] = df[col].astype('int32')
        df['sender_txn_count_prev'] = df['sender_txn_count_prev'].astype('int64')
        double_cols = ['amount_norm', 'txn_velocity', 'sender_avg_amount_prev', 'amount_weekend', 
                       'amount_vs_sender_mean', 'amount_hour_interaction', 'amount_inr', 
                       'sender_max_amount_prev', 'amount_ratio_deviation']
        for col in double_cols:
            df[col] = df[col].astype('float64')
        pred = model.predict(df)
        if isinstance(pred, pd.DataFrame):
            return {"success": True, "fraud_prob": float(pred["fraud_probability"].iloc[0]), "fraud_pred": int(pred["fraud_prediction"].iloc[0])}
        elif isinstance(pred, np.ndarray):
            prob = float(pred[0][0] if len(pred[0]) > 0 else pred[0])
            return {"success": True, "fraud_prob": prob, "fraud_pred": int(prob > 0.1)}
        else:
            prob = float(pred)
            return {"success": True, "fraud_prob": prob, "fraud_pred": int(prob > 0.1)}
    except Exception as e:
        return {"success": False, "error": str(e)}

def detect_language(text):
    if any('\u0900' <= c <= '\u097F' for c in text): return "Hindi"
    elif any('\u0B80' <= c <= '\u0BFF' for c in text): return "Tamil"
    elif any('\u0C00' <= c <= '\u0C7F' for c in text): return "Telugu"
    elif any('\u0980' <= c <= '\u09FF' for c in text): return "Bengali"
    elif any('\u0A80' <= c <= '\u0AFF' for c in text): return "Gujarati"
    elif any('\u0C80' <= c <= '\u0CFF' for c in text): return "Kannada"
    elif any('\u0D00' <= c <= '\u0D7F' for c in text): return "Malayalam"
    elif any('\u0A00' <= c <= '\u0A7F' for c in text): return "Punjabi"
    else: return "English"

def generate_text_rag(prompt, model, tokenizer, max_tokens=150):
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    except Exception as e:
        return f"Error: {str(e)}"

def simple_rag_query(question, kb, embed_model, llm, tokenizer):
    try:
        from numpy import dot
        from numpy.linalg import norm
        lang = detect_language(question)
        q_emb = embed_model.encode([question])
        kb_texts = [d["text"] for d in kb]
        kb_embs = embed_model.encode(kb_texts)
        sims = [dot(q_emb[0], k) / (norm(q_emb[0]) * norm(k)) for k in kb_embs]
        top_idx = np.argsort(sims)[-2:][::-1]
        context = "\n\n".join([kb[i]["text"] for i in top_idx])
        
        # Stronger language-specific prompts in native script
        lang_prompts = {
            "English": f"Context: {context}\n\nQuestion: {question}\n\nIMPORTANT: You must answer ONLY in English. Do not use any other language.\n\nAnswer:",
            "Hindi": f"संदर्भ: {context}\n\nप्रश्न: {question}\n\nमहत्वपूर्ण: आपको केवल हिंदी में उत्तर देना है। किसी अन्य भाषा का उपयोग न करें।\n\nउत्तर:",
            "Tamil": f"சூழல்: {context}\n\nகேள்வி: {question}\n\nமுக்கியம்: நீங்கள் தமிழில் மட்டுமே பதிலளிக்க வேண்டும். வேறு எந்த மொழியையும் பயன்படுத்த வேண்டாம்.\n\nபதில்:",
            "Telugu": f"సందర్భం: {context}\n\nప్రశ్న: {question}\n\nముఖ్యమైనది: మీరు తెలుగులో మాత్రమే సమాధానం ఇవ్వాలి. మరే ఇతర భాషను ఉపయోగించవద్దు.\n\nసమాధానం:",
            "Bengali": f"প্রসঙ্গ: {context}\n\nপ্রশ্ন: {question}\n\nগুরুত্বপূর্ণ: আপনাকে শুধুমাত্র বাংলায় উত্তর দিতে হবে। অন্য কোন ভাষা ব্যবহার করবেন না।\n\nউত্তর:",
            "Marathi": f"संदर्भ: {context}\n\nप्रश्न: {question}\n\nमहत्त्वाचे: तुम्ही फक्त मराठीत उत्तर द्यावे। इतर कोणतीही भाषा वापरू नका।\n\nउत्तर:",
            "Gujarati": f"સંદર્ભ: {context}\n\nપ્રશ્ન: {question}\n\nમહત્વપૂર્ણ: તમારે ફક્ત ગુજરાતીમાં જવાબ આપવો જોઈએ. અન્ય કોઈ ભાષાનો ઉપયોગ કરશો નહીં.\n\nજવાબ:",
            "Kannada": f"ಸಂದರ್ಭ: {context}\n\nಪ್ರಶ್ನೆ: {question}\n\nಮಹತ್ವದ: ನೀವು ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಉತ್ತರಿಸಬೇಕು. ಬೇರೆ ಯಾವುದೇ ಭಾಷೆಯನ್ನು ಬಳಸಬೇಡಿ.\n\nಉತ್ತರ:",
            "Malayalam": f"സന്ദർഭം: {context}\n\nചോദ്യം: {question}\n\nപ്രധാനം: നിങ്ങൾ മലയാളത്തിൽ മാത്രം ഉത്തരം നൽകണം. മറ്റ് ഭാഷകൾ ഉപയോഗിക്കരുത്.\n\nഉത്തരം:",
            "Punjabi": f"ਸੰਦਰਭ: {context}\n\nਸਵਾਲ: {question}\n\nਮਹੱਤਵਪੂਰਨ: ਤੁਹਾਨੂੰ ਸਿਰਫ਼ ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦੇਣਾ ਚਾਹੀਦਾ ਹੈ। ਕੋਈ ਹੋਰ ਭਾਸ਼ਾ ਨਾ ਵਰਤੋ।\n\nਜਵਾਬ:"
        }
        
        prompt = lang_prompts.get(lang, f"Context: {context}\n\nQuestion: {question}\n\nIMPORTANT: Answer ONLY in English.\n\nAnswer:")
        answer = generate_text_rag(prompt, llm, tokenizer, 200)
        
        # Post-process: if answer is in wrong language, try again with stricter prompt
        detected_answer_lang = detect_language(answer)
        if detected_answer_lang != lang and lang != "English":
            # Retry with even stronger instruction
            retry_prompt = f"{prompt}\n\n[{lang} भाषा में ही जवाब दें]"
            answer = generate_text_rag(retry_prompt, llm, tokenizer, 200)
        
        return {"success": True, "answer": answer, "detected_language": lang}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Load models
fraud_model = load_fraud_model()
loan_model = load_loan_model()

st.title("Rakshak - Financial Intelligence Platform")
st.markdown("Real-time ML-powered Fraud Detection | Credit Analysis | Multilingual AI Assistant")

with st.sidebar:
    st.header("Model Status")
    st.success("Fraud: Loaded" if fraud_model else "Fraud: Failed")
    st.success("Loan: Loaded" if loan_model else "Loan: Failed")
    st.markdown("---")
    st.text("Inference: In-App")
    st.text("Memory: ~15GB")

tab1, tab2, tab3 = st.tabs(["Fraud Detection", "Credit Eligibility", "AI Assistant"])

# TAB 1: FRAUD DETECTION
with tab1:
    st.header("UPI Fraud Detection")
    st.markdown("Analyze transactions for fraud using XGBoost + DQN model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Transaction Info")
        amount_inr = st.number_input("Amount (INR)", min_value=0.0, value=3674.69, step=100.0)
        txn_type = st.selectbox("Transaction Type", ["P2P", "P2M", "Bill Payment", "Recharge"], index=3)
        txn_type_enc = ["P2P", "P2M", "Bill Payment", "Recharge"].index(txn_type)
        cat = st.selectbox("Category", ["Food", "Retail", "Services"], index=2)
        cat_enc = ["Food", "Retail", "Services"].index(cat)
        device = st.selectbox("Device", ["Android", "iOS", "Web"], index=2)
        device_enc = ["Android", "iOS", "Web"].index(device)
    
    with col2:
        st.subheader("Time Details")
        hour_of_day = st.slider("Hour of Day", 0, 23, 3)
        day_of_week = st.slider("Day (0=Mon, 6=Sun)", 0, 6, 0)
        is_weekend = st.checkbox("Is Weekend", value=False)
        network = st.selectbox("Network", ["WiFi", "4G", "5G"], index=0)
        network_enc = ["WiFi", "4G", "5G"].index(network)
    
    with col3:
        st.subheader("User Details")
        sender_age = st.selectbox("Sender Age", ["18-25", "26-35", "36-45", "46-60", "60+"], index=1)
        sender_age_enc = ["18-25", "26-35", "36-45", "46-60", "60+"].index(sender_age)
        receiver_age = st.selectbox("Receiver Age", ["18-25", "26-35", "36-45", "46-60", "60+"], index=0)
        receiver_age_enc = ["18-25", "26-35", "36-45", "46-60", "60+"].index(receiver_age)
        sender_avg = st.number_input("Sender Avg Amount", value=4303.93, step=100.0)
        sender_txn_count = st.number_input("Sender Txn Count", value=77, step=1, min_value=0)
    
    if st.button("Analyze Transaction", type="primary", use_container_width=True):
        if fraud_model:
            # Calculate derived features
            amount_norm = amount_inr / 50000.0
            amount_vs_sender_mean = amount_inr / sender_avg if sender_avg > 0 else 0
            odd_hour_flag = 1 if hour_of_day < 6 or hour_of_day > 22 else 0
            high_amount_flag = 1 if amount_inr > 10000 else 0
            
            txn_data = {
                'day_of_week': day_of_week,
                'device_enc': device_enc,
                'amount_norm': amount_norm,
                'hour_of_day': hour_of_day,
                'txn_velocity': 0.0,
                'odd_hour_flag': odd_hour_flag,
                'device_network_interaction': device_enc * 100 + network_enc,
                'sender_avg_amount_prev': sender_avg,
                'amount_weekend': amount_inr * (1 if is_weekend else 0),
                'fuel_large_flag': 1 if cat_enc == 2 and amount_inr > 5000 else 0,
                'high_amount_odd_hour': high_amount_flag * odd_hour_flag,
                'cat_device_interaction': cat_enc * 10 + device_enc,
                'same_bank_flag': 1,
                'sender_age_enc': sender_age_enc,
                'weekend_high_spend': (1 if is_weekend else 0) * high_amount_flag,
                'txn_type_enc': txn_type_enc,
                'high_amount_flag': high_amount_flag,
                'is_weekend': 1 if is_weekend else 0,
                'receiver_age_enc': receiver_age_enc,
                'amount_vs_sender_mean': amount_vs_sender_mean,
                'amount_hour_interaction': amount_norm * hour_of_day,
                'network_enc': network_enc,
                'cat_enc': cat_enc,
                'amount_inr': amount_inr,
                'sender_max_amount_prev': sender_avg * 1.5,
                'amount_ratio_deviation': abs(amount_vs_sender_mean - 1.0),
                'sender_txn_count_prev': sender_txn_count
            }
            
            with st.spinner("Analyzing..."):
                result = predict_fraud(txn_data, fraud_model)
            
            if result["success"]:
                prob = result["fraud_prob"]
                pred = result["fraud_pred"]
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Fraud Probability", f"{prob*100:.2f}%")
                with col_b:
                    risk = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"
                    color = "#ff4444" if prob >= 0.7 else "#ffaa00" if prob >= 0.4 else "#44ff44"
                    st.markdown(f"**Risk Level:** <span style='color:{color};font-size:1.5em;font-weight:bold'>{risk}</span>", unsafe_allow_html=True)
                with col_c:
                    st.metric("Prediction", "FRAUD" if pred == 1 else "LEGITIMATE")
                
                if pred == 1:
                    st.error("ALERT: This transaction is flagged as fraudulent!")
                else:
                    st.success("This transaction appears legitimate.")
            else:
                st.error(f"Error: {result['error']}")
        else:
            st.error("Fraud model not loaded!")

# TAB 2: CREDIT ELIGIBILITY
with tab2:
    st.header("Credit Eligibility Assessment")
    st.markdown("Evaluate applicant based on banking behavior and transaction patterns")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Transaction Patterns")
        monthly_inflow_avg = st.number_input("Monthly Inflow Avg (INR)", min_value=0.0, value=45000.0, step=1000.0)
        avg_transaction_amount = st.number_input("Avg Transaction Amount (INR)", min_value=0.0, value=2500.0, step=100.0)
        max_transaction = st.number_input("Max Transaction (INR)", min_value=0.0, value=15000.0, step=1000.0)
        transaction_velocity = st.number_input("Transaction Velocity (txns/day)", min_value=0.0, value=3.5, step=0.1)
    
    with col2:
        st.subheader("Transaction Ratios")
        large_txn_ratio = st.slider("Large Txn Ratio", 0.0, 1.0, 0.15, step=0.01)
        small_txn_ratio = st.slider("Small Txn Ratio", 0.0, 1.0, 0.45, step=0.01)
        weekend_txn_ratio = st.slider("Weekend Txn Ratio", 0.0, 1.0, 0.25, step=0.01)
        night_txn_ratio = st.slider("Night Txn Ratio", 0.0, 1.0, 0.10, step=0.01)
    
    with col3:
        st.subheader("Account Health")
        bounce_rate = st.slider("Bounce Rate", 0.0, 1.0, 0.02, step=0.01)
        credit_debit_ratio = st.slider("Credit/Debit Ratio", 0.0, 5.0, 1.2, step=0.1)
        transaction_amount_volatility = st.number_input("Amount Volatility", min_value=0.0, value=0.35, step=0.01)
        active_days = st.number_input("Active Days (last 30)", min_value=0, value=25, step=1)
    
    if st.button("Assess Eligibility", type="primary", use_container_width=True):
        if loan_model:
            # Banking behavior features matching model schema
            applicant = {
                'monthly_inflow_avg': float(monthly_inflow_avg),
                'transaction_velocity': float(transaction_velocity),
                'bounce_rate': float(bounce_rate),
                'avg_transaction_amount': float(avg_transaction_amount),
                'transaction_amount_volatility': float(transaction_amount_volatility),
                'large_txn_ratio': float(large_txn_ratio),
                'max_transaction': float(max_transaction),
                'weekend_txn_ratio': float(weekend_txn_ratio),
                'night_txn_ratio': float(night_txn_ratio),
                'credit_debit_ratio': float(credit_debit_ratio),
                'active_days': float(active_days),
                'small_txn_ratio': float(small_txn_ratio)
            }
            
            with st.spinner("Evaluating banking behavior..."):
                try:
                    df = pd.DataFrame([applicant])
                    result = loan_model.predict(df)
                    
                    if isinstance(result, pd.DataFrame):
                        prediction = result.to_dict(orient="records")[0]
                    elif isinstance(result, np.ndarray):
                        prediction = {"eligibility": float(result[0])}
                    else:
                        prediction = {"eligibility": result}
                    
                    st.success("Assessment Complete!")
                    
                    # Display results
                    elig_score = prediction.get("eligibility", prediction.get("prediction", 0.5))
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Eligibility Score", f"{elig_score*100:.1f}%")
                    with col_b:
                        status = "APPROVED" if elig_score > 0.6 else "NEEDS REVIEW" if elig_score > 0.4 else "REJECTED"
                        color = "#44ff44" if elig_score > 0.6 else "#ffaa00" if elig_score > 0.4 else "#ff4444"
                        st.markdown(f"**Status:** <span style='color:{color};font-size:1.5em;font-weight:bold'>{status}</span>", unsafe_allow_html=True)
                    
                    if elig_score > 0.6:
                        st.success("Applicant shows strong banking behavior - ELIGIBLE!")
                    elif elig_score > 0.4:
                        st.warning("Banking behavior is acceptable - needs additional review.")
                    else:
                        st.error("Banking behavior shows high risk - NOT ELIGIBLE.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.error("Loan model not loaded!")

# TAB 3: AI ASSISTANT
with tab3:
    st.header("Multilingual AI Assistant (Sarvam-1)")
    st.markdown("Ask questions in English, Hindi, Tamil, Telugu, Bengali, and more!")
    
    # Load RAG button
    if 'rag_models' not in st.session_state:
        st.info("Click below to load the RAG models (takes 5-10 min first time)")
        if st.button("Load RAG Models", type="primary"):
            with st.spinner("Loading Sarvam-1 and embeddings..."):
                rag = load_rag_models()
                if rag.get("success"):
                    st.session_state['rag_models'] = rag
                    st.success("RAG models loaded successfully!")
                    st.rerun()
                else:
                    st.error(f"Failed to load: {rag.get('error')}")
    
    if 'rag_models' in st.session_state:
        rag = st.session_state['rag_models']
        
        st.success("RAG Models Loaded!")
        
        # Knowledge base
        kb = [
            {"text": "UPI (Unified Payments Interface) is India's instant real-time payment system developed by NPCI. It allows instant money transfer between bank accounts through mobile devices.", "language": "English"},
            {"text": "Fraud detection uses machine learning algorithms like XGBoost and Deep Q-Networks to identify suspicious transactions in real-time based on transaction patterns, user behavior, and risk indicators.", "language": "English"},
            {"text": "यूपीआई (एकीकृत भुगतान इंटरफ़ेस) भारत की तत्काल भुगतान प्रणाली है जो एनपीसीआई द्वारा विकसित की गई है। यह मोबाइल उपकरणों के माध्यम से बैंक खातों के बीच तुरंत पैसे ट्रांसफर करने की अनुमति देता है।", "language": "Hindi"},
            {"text": "धोखाधड़ी का पता लगाने के लिए मशीन लर्निंग का उपयोग किया जाता है। यह लेनदेन पैटर्न, उपयोगकर्ता व्यवहार और जोखिम संकेतकों के आधार पर संदिग्ध लेनदेन की पहचान करता है।", "language": "Hindi"},
            {"text": "UPI (ஒருங்கிணைக்கப்பட்ட பணம் செலுத்தும் இடைமுகம்) என்பது இந்தியாவின் உடனடி நேரடி பணம் செலுத்தும் முறையாகும். இது மொபைல் சாதனங்கள் மூலம் வங்கி கணக்குகளுக்கு இடையே உடனடியாக பணம் மாற்ற அனுமதிக்கிறது.", "language": "Tamil"},
            {"text": "மோசடி கண்டறிதல் இயந்திர கற்றல் வழிமுறைகளைப் பயன்படுத்துகிறது. இது பரிவர்த்தனை முறைகள், பயனர் நடத்தை மற்றும் ஆபத்து குறிகாட்டிகளின் அடிப்படையில் சந்தேகத்திற்கிடமான பரிவர்த்தனைகளை அடையாளம் காண்கிறது.", "language": "Tamil"},
            {"text": "UPI (ఏకీకృత చెల్లింపుల ఇంటర్‌ఫేస్) అనేది NPCI చే అభివృద్ధి చేయబడిన భారతదేశం యొక్క తక్షణ నిజ-సమయ చెలున్ల వ్యవస్థ. ఇది మొబైల్ పరికరాల ద్వారా బ్యాంక్ ఖాతాల మధ్య తక్షణ డబ్బు బదిలీని అనుమతిస్తుంది.", "language": "Telugu"},
            {"text": "మోసం గుర్తింపు మెషిన్ లెర్నింగ్ అల్గారిథమ్‌లను ఉపయోగిస్తుంది. ఇది లావాదేవీ నమూనాలు, వినియోగదారు ప్రవర్తన మరియు ప్రమాద సూచికల ఆధారంగా అనుమానాస్పద లావాదేవీలను గుర్తిస్తుంది.", "language": "Telugu"},
            {"text": "UPI (ইউনিফাইড পেমেন্টস ইন্টারফেস) হল ভারতের তাৎক্ষণিক রিয়েল-টাইম পেমেন্ট সিস্টেম যা NPCI দ্বারা উন্নত। এটি মোবাইল ডিভাইসের মাধ্যমে ব্যাংক অ্যাকাউন্টের মধ্যে তাৎক্ষণিক অর্থ স্থানান্তর করতে দেয়।", "language": "Bengali"},
            {"text": "জালিয়াতি সনাক্তকরণ মেশিন লার্নিং অ্যালগরিদম ব্যবহার করে। এটি লেনদেন প্যাটার্ন, ব্যবহারকারীর আচরণ এবং ঝুঁকি সূচকের উপর ভিত্তি করে সন্দেহজনক লেনদেন চিহ্নিত করে।", "language": "Bengali"},
            {"text": "Credit eligibility is determined by factors like credit score, income stability, debt-to-income ratio, employment history, and collateral. Banks use ML models to assess risk and approve loans.", "language": "English"}
        ]
        
        # Sample questions by language
        st.subheader("Sample Questions")
        
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            st.markdown("**English:**")
            if st.button("What is UPI?"):
                st.session_state['question'] = "What is UPI?"
            if st.button("How does fraud detection work?"):
                st.session_state['question'] = "How does fraud detection work?"
            
            st.markdown("**Hindi (हिंदी):**")
            if st.button("यूपीआई क्या है?"):
                st.session_state['question'] = "यूपीआई क्या है?"
            if st.button("धोखाधड़ी का पता कैसे लगाया जाता है?"):
                st.session_state['question'] = "धोखाधड़ी का पता कैसे लगाया जाता है?"
        
        with col_q2:
            st.markdown("**Tamil (தமிழ்):**")
            if st.button("UPI என்றால் என்ன?"):
                st.session_state['question'] = "UPI என்றால் என்ன?"
            if st.button("மோசடி கண்டறிதல் எப்படி வேலை செய்கிறது?"):
                st.session_state['question'] = "மோசடி கண்டறிதல் எப்படி வேலை செய்கிறது?"
            
            st.markdown("**Telugu (తెలుగు):**")
            if st.button("UPI అంటే ఏమిటి?"):
                st.session_state['question'] = "UPI అంటే ఏమిటి?"
            if st.button("మోసం గుర్తింపు ఎలా పనిచేస్తుంది?"):
                st.session_state['question'] = "మోసం గుర్తింపు ఎలా పనిచేస్తుంది?"
        
        st.markdown("---")
        
        # Question input
        question = st.text_area(
            "Your Question (any language):", 
            value=st.session_state.get('question', ''),
            height=100,
            placeholder="Type your question in English, Hindi, Tamil, Telugu, Bengali, or any supported language..."
        )
        
        if st.button("Get Answer", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("Generating answer..."):
                    result = simple_rag_query(question, kb, rag['embedding_model'], rag['llm_model'], rag['llm_tokenizer'])
                
                if result['success']:
                    st.success(f"**Detected Language:** {result['detected_language']}")
                    
                    st.markdown("### Answer:")
                    st.info(result['answer'])
                    
                else:
                    st.error(f"Error: {result['error']}")
            else:
                st.warning("Please enter a question!")
        
        # Clear button
        if st.button("Clear"):
            st.session_state['question'] = ''
            st.rerun()

st.markdown("---")
st.caption("Rakshak Financial Intelligence Platform | Fraud Detection | Credit Eligibility | Multilingual AI Assistant")
