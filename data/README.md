# 💰 Loan Default Predictor (End-to-End ML Project)

This repository contains a complete, production-ready **End-to-End Machine Learning system** for predicting the probability of default on credit card payments using the **UCI Default of Credit Card Clients Dataset**.

The project demonstrates:

* Reproducible ML pipeline
* Data ingestion, cleaning, preprocessing
* Model training with LightGBM
* Explainability (SHAP, LIME optional)
* Deployment via **Streamlit UI** and **FastAPI**
* Optional multi-model support, dark mode UI, admin logs, authentication
* Clear modular code under `src/`

---

# 📁 Project Structure

```
loan-default-project/
├── data/
│   ├── raw/                     # Raw Excel/CSV files (ignored in Git)
│   └── processed/               # Cleaned CSV used for training
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── 01_data_overview.ipynb
│   └── 02_feature_engineering.ipynb
│
├── src/
│   ├── data/
│   │   └── load_data.py         # Downloading, validating & loading dataset
│   │
│   ├── features/
│   │   └── build_features.py    # Preprocessing pipeline (ColumnTransformer)
│   │
│   ├── models/
│   │   └── train.py             # Training pipeline (LightGBM + metrics)
│   │
│   ├── inference/
│   │   └── api.py               # FastAPI prediction service
│   │
│   └── ui/
│       └── streamlit_app.py     # Streamlit prediction + explainability UI
│
├── models/                      # Saved model artifacts
│   ├── final_model.joblib
│   ├── feature_metadata.json
│   └── preprocessors/
│       └── preprocessor.joblib
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

# 🔄 ML Pipeline Diagram

```
          ┌──────────────────────┐
          │     Data Source      │
          │  UCI Default Dataset │
          └──────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  load_data.py              │
        │ - download if missing      │
        │ - convert XLS → CSV        │
        │ - clean columns            │
        └──────────┬─────────────────┘
                    │ DataFrame
                    ▼
        ┌────────────────────────────┐
        │ build_features.py          │
        │ - numeric imputer/scaler   │
        │ - categorical encoder      │
        │ - ColumnTransformer        │
        └──────────┬─────────────────┘
                    │ Preprocessed X
                    ▼
        ┌────────────────────────────┐
        │ train.py (LightGBM)        │
        │ - train/val/test split     │
        │ - ROC-AUC, PR-AUC, acc     │
        │ - save model + preprocessor│
        └──────────┬─────────────────┘
                    │ joblib artifacts
                    ▼
        ┌────────────────────────────┐
        │ Deployment (FastAPI)       │
        │   /predict endpoint        │
        └──────────┬─────────────────┘
                    │
                    ▼
        ┌────────────────────────────┐
        │ Streamlit UI               │
        │ - manual input / CSV       │
        │ - shap explainability      │
        │ - metrics dashboard        │
        └────────────────────────────┘
```

---

# ⚙️ Setup Guide

## 1️⃣ Create a virtual environment

```
python -m venv env
source env/bin/activate    # Linux / Mac
env\Scripts\activate       # Windows
```

## 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

# 📊 Data Preparation

### **Run Notebook 01 to download + clean dataset**

Creates:

```
data/processed/default_of_credit_card_clients.csv
```

### **Run Notebook 02 for preprocessing pipeline**

Creates:

```
models/preprocessors/preprocessor.joblib
```

---

# 🏋️ Train the Final Model

Use module execution so imports work:

```
python -m src.models.train
```

Artifacts generated:

```
models/final_model.joblib
models/preprocessors/preprocessor.joblib
models/feature_metadata.json
```

---

# 🚀 Running the FastAPI Inference Server

```
uvicorn src.inference.api:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger Docs:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

---

# 🔮 Example Prediction Payload

Send to:

```
POST http://localhost:8000/predict
```

Payload:

```json
{
  "LIMIT_BAL": 20000.0,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 24,
  "PAY_0": 2,
  "PAY_2": 2,
  "PAY_3": -1,
  "PAY_4": -1,
  "PAY_5": -1,
  "PAY_6": -1,
  "BILL_AMT1": 3913.0,
  "BILL_AMT2": 3102.0,
  "BILL_AMT3": 689.0,
  "BILL_AMT4": 0.0,
  "BILL_AMT5": 0.0,
  "BILL_AMT6": 0.0,
  "PAY_AMT1": 0.0,
  "PAY_AMT2": 689.0,
  "PAY_AMT3": 0.0,
  "PAY_AMT4": 0.0,
  "PAY_AMT5": 0.0,
  "PAY_AMT6": 0.0
}
```

Response includes:

* `probability_of_default`
* SHAP top feature explanation

---

# 🎨 Streamlit UI (Manual Input + CSV + SHAP)

Run the app:

```
streamlit run src/ui/streamlit_app.py
```

Features:

* Upload CSV for batch predictions
* Manual input form with dropdowns
* SHAP force plot + beeswarm
* Model metrics dashboard
* **Dark mode toggle**
* **Multi-model switching** (if enabled)
* **Admin logs**
* **Authentication (optional)**

---

# 🔐 Authentication (Optional)

Streamlit secrets file:

```
.streamlit/secrets.toml
```

Example:

```
[auth]
username = "admin"
password = "yourpassword123"
```

Used inside Streamlit for login gating.

---

```

---

# 🧹 In-app Data Cleaning UI

The Streamlit UI optionally supports:

* dropping rows
* removing outliers
* replacing missing values
* renaming columns

Results feed back into the prediction pipeline.

---

# 🧪 Tests (optional)

Recommended structure:

```
tests/
  ├── test_data.py
  ├── test_features.py
  ├── test_train.py
  └── test_api.py
```

---

# 📦 Deployment Options

* **Local FastAPI + Streamlit**


---

# 🙌 Contributing

Pull requests welcome!

---


