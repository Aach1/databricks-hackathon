# Rakshak - Financial Intelligence Platform

🛡️ **Real-time ML-powered Fraud Detection | Credit Analysis | Multilingual AI Assistant**

## 🎯 Overview

Rakshak is an AI-powered financial intelligence platform built on Databricks that provides:

1. **Fraud Detection** - Real-time UPI transaction fraud detection using XGBoost + DQN hybrid model
2. **Credit Eligibility** - Banking behavior-based loan eligibility assessment
3. **Multilingual AI Assistant** - RAG pipeline powered by Sarvam-1 (7B) LLM supporting 10+ Indian languages

## 🏗️ Tech Stack

### Platform & Infrastructure
- **Databricks Apps** - Hosting and deployment
- **Serverless Compute** - Auto-scaling (~15GB RAM)

### Web Framework
- **Streamlit** - Python web framework for interactive UI

### Machine Learning & AI
- **XGBoost + DQN** - Hybrid fraud detection model
- **Banking Behavior Classifier** - Credit eligibility model
- **Sarvam-1 (7B)** - Indian multilingual LLM
- **MLflow** - Model management and serving
- **PyTorch** - Deep learning framework

### NLP & Embeddings
- **Hugging Face Transformers** - Model loading
- **Sentence Transformers** - Multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
- **FAISS** - Vector similarity search

### Model Optimization
- **BitsAndBytes** - 8-bit quantization (reduced Sarvam-1 from ~14GB to ~3.5GB)
- **Accelerate** - Optimized model loading

## 🌐 Languages Supported

English, Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi

## 📊 Memory Footprint

- Fraud Model: ~50MB
- Credit Model: ~10MB
- Sarvam-1 (8-bit): ~3.5GB
- Embeddings: ~470MB
- **Total: ~4GB** (CPU-optimized for hackathon constraints)

## 🚀 Features

### Tab 1: Fraud Detection
- Real-time UPI transaction analysis
- 27 engineered features (transaction patterns, time-based, user behavior)
- Color-coded risk levels (HIGH/MEDIUM/LOW)
- Fraud probability scoring

### Tab 2: Credit Eligibility
- Banking behavior assessment
- 12 transaction-based features (velocity, ratios, account health)
- Eligibility scoring and status (APPROVED/NEEDS REVIEW/REJECTED)

### Tab 3: Multilingual AI Assistant
- Retrieval-Augmented Generation (RAG) pipeline
- Knowledge base with 11 documents in 6 languages
- Automatic language detection
- Same-language response guarantee
- Sample questions in multiple languages

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── app.yaml               # App configuration
├── app_old.py             # Backup version
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Databricks workspace (for deployment)

### Dependencies

```bash
pip install -r requirements.txt
```

## 🏃 Running Locally

```bash
streamlit run app.py
```

## 📦 Deployment on Databricks

1. Upload files to Databricks workspace
2. Create a Databricks App
3. Point to the app directory
4. App will auto-deploy with serverless compute

## 🔑 Model URIs

- Fraud Model: `runs:/cabe6e4aac7f42bc80e46f3fa402e885/hybrid_fraud_model`
- Credit Model: `runs:/81848d9871fc49c7987e233bced963eb/model`
- Sarvam-1: `sarvamai/sarvam-1`
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2`

## 🎓 Built For

Digital Artha Hackathon 2026

## 👥 Team

[Add your team name and members here]

## 📄 License

[Add license information]

## 🙏 Acknowledgments

- Databricks for the platform
- Sarvam AI for the Sarvam-1 LLM
- Hugging Face for model hosting
