# Rakshak-Artha: Fraud Detection & Credit Scoring for Rural India 🛡️💰

> Protecting wealth in real-time and unlocking micro-credit through digital footprints — communicated in native languages.

A Databricks-hosted Streamlit app with three tools: a UPI fraud-risk scorer, a banking-behavior credit-eligibility scorer, and a multilingual AI assistant that answers questions about UPI and fraud in 10+ Indian languages.

**Live demo:** https://digital-artha-sarvam-7474643766841203.aws.databricksapps.com/ *(Databricks Apps compute can idle-stop — give it a minute to spin up if it doesn't load immediately)*

---

## 🎯 Motivation

Millions of rural users are entering the digital economy through UPI, but they face a dual problem:
- **Vulnerability**: exposed to digital financial fraud with no real-time protection
- **Exclusion**: locked out of formal credit due to lack of a traditional CIBIL score

Rakshak-Artha explores both sides on one Databricks-native stack: score transactions for fraud risk, score a user's payment behavior for credit eligibility, and let people ask questions about either in their own language.

---

## ⚙️ What It Does

The app is a single Streamlit UI (`src/app/app.py`) with three tabs:

### 1. Fraud Detection
A form for a UPI transaction (amount, type, category, device, time, sender/receiver profile). On submit, the app derives ~26 engineered features (odd-hour flag, amount-vs-sender-mean ratio, weekend/high-amount interactions, etc.) and scores them against an **XGBoost** model loaded from MLflow, returning a fraud probability and a Low/Medium/High risk band.

### 2. Credit Eligibility
A form for 12 banking-behavior signals (monthly inflow, transaction velocity, bounce rate, large/small/weekend/night transaction ratios, credit-debit ratio, active days). Scores them against a separately-trained MLflow model and returns an eligibility score with an Approved / Needs Review / Rejected verdict.

### 3. Multilingual AI Assistant
A small RAG chatbot over **Sarvam-1** (7B), loaded 8-bit quantized (BitsAndBytes) so it fits in ~3.5GB instead of ~14GB. It embeds the question with a multilingual sentence-transformer (`paraphrase-multilingual-MiniLM-L12-v2`), retrieves the closest matches from a small in-app knowledge base about UPI and fraud detection, detects the question's language from its Unicode script (English, Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi), and prompts Sarvam-1 to answer in that same language.

---

## 🛠️ How It's Built

### Data & training (`src/jobs/`)
- **`data_gen.py`** — a PySpark job that generates 100k synthetic UPI-style transactions (amount, category, timestamp, a rule-based fraud label skewed by amount/category) and writes them to a Unity Catalog Delta table.
- **`train.py`** — reads that table, trains an **XGBoost** classifier and registers it to Unity Catalog via MLflow. It then uses the XGBoost probability as part of the state for a custom Gym environment and trains a **DQN** agent (`stable-baselines3`) with a 3-action policy (Allow / Review / Block) and reward shaping (blocking real fraud is rewarded, allowing fraud is penalized heavily), logging it as an MLflow artifact. This is the "hybrid XGBoost + DQN" model referenced throughout the app — the DQN training job is included, but the live app currently scores transactions with XGBoost directly (see [Current Limitations](#-current-limitations-honest-scope)).
- The credit-eligibility model is trained separately and loaded by MLflow run ID in the app; its training script isn't in this repo yet.

### Serving (`src/app/`)
`app.py` loads both models in-process via `mlflow.pyfunc.load_model("runs:/<run_id>/...")` and runs inference directly inside the Streamlit app — there's no separate model-serving call in the current UI.

### Deployment — Databricks Asset Bundle
The whole thing ships as a **Databricks Asset Bundle** (`databricks.yml`):
- `resources/app.yml` — deploys the Streamlit app in `src/app/` as a Databricks App.
- `resources/jobs.yml` — defines two jobs, `data_generation` and `model_training` (which depends on it), each running on its own job cluster.

### Testing
`src/tests/test_endpoints.py` is a standalone script for smoke-testing Databricks **Model Serving** endpoints directly over HTTP (separately from the Streamlit app, which calls MLflow in-process rather than a serving endpoint).

---

## 📁 Repository Structure

```
databricks-hackathon/
├── databricks.yml          # Databricks Asset Bundle definition
├── resources/
│   ├── app.yml              # Databricks App resource (Streamlit frontend)
│   └── jobs.yml              # Jobs: synthetic data generation + model training
└── src/
    ├── app/                  # Streamlit frontend (fraud + credit + AI assistant tabs)
    │   ├── app.py
    │   ├── app.yaml
    │   └── requirements.txt
    ├── jobs/                 # PySpark data generation + XGBoost/DQN training
    │   ├── data_gen.py
    │   └── train.py
    └── tests/
        └── test_endpoints.py # Manual smoke test for the model-serving endpoints
```

---

## 🧰 Tech Stack

| Area | Tools |
|---|---|
| Data | PySpark, Delta Lake, Unity Catalog |
| Fraud model | XGBoost |
| Experimental decision layer | Deep Q-Network via `stable-baselines3` + `gymnasium` |
| Model tracking/registry | MLflow |
| LLM assistant | Sarvam-1 (7B) via Hugging Face `transformers`, 8-bit via BitsAndBytes, `accelerate` |
| Retrieval | `sentence-transformers` (multilingual MiniLM), cosine similarity |
| Frontend | Streamlit, deployed as a Databricks App |
| Packaging | Databricks Asset Bundles (`databricks.yml`) |

---

## 🚀 Running It

**Deploy to Databricks** (requires the Databricks CLI configured with a workspace profile):
```bash
databricks bundle deploy -t dev
databricks bundle run data_generation -t dev
databricks bundle run model_training -t dev
```
This provisions the Streamlit app resource and runs the two jobs. Update the hardcoded `runs:/<run_id>/...` model URIs in `app.py` to point at the run IDs your `model_training` job produces, and register/point at your own credit-eligibility model similarly.

**Run the UI locally** (after training/registering the models):
```bash
cd src/app
pip install -r requirements.txt
streamlit run app.py
```
Needs `mlflow.set_tracking_uri("databricks")` credentials available in the environment (e.g. `DATABRICKS_HOST` / `DATABRICKS_TOKEN`).

---

## ⚠️ Current Limitations (honest scope)

This is a hackathon build — a few things are simplified or aspirational rather than fully wired up:
- **DQN isn't in the live scoring path yet.** It's trained in `train.py` and logged to MLflow, but `app.py` currently scores transactions with XGBoost alone.
- **No medallion (bronze/silver/gold) pipeline yet.** `data_gen.py` writes one synthetic Delta table; there's no ingestion layer for real UPI feeds or a bronze→silver→gold structure yet.
- **The AI assistant's knowledge base is a small in-app demo set** (~11 facts about UPI/fraud), not a full indexed corpus of RBI documents. Retrieval is plain cosine similarity over those entries — `faiss-cpu` is in `requirements.txt` for a real vector index, but isn't used yet.
- **No SMS delivery.** The assistant's answers are shown in the Streamlit UI only.
- **The credit-eligibility model's training code isn't in this repo** — the app loads it by MLflow run ID, but reproducing it from scratch isn't yet possible from this codebase.

---

## 🎯 Roadmap

- Wire the trained DQN policy into `app.py`'s live fraud-scoring path
- Replace the in-memory knowledge base with a FAISS index over real RBI guideline documents
- Add a training job for the credit-eligibility model, matching the fraud pipeline
- Real UPI-feed ingestion with a proper bronze → silver → gold Delta pipeline
- SMS/notification delivery for fraud alerts and RAG explanations
- Voice input for disputes (speech-to-text into the Sarvam-1 pipeline)

---

## 👥 Team

Abhiraj Kumar
Harshith Jay Surya Ganji

---

## 📝 License

No license has been chosen yet for this project.
