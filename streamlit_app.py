"""
streamlit_app_improved.py

Full-featured Streamlit UI for:
- Dataset preview & examples gallery
- Batch CSV upload with validation
- Manual input form (smart widgets + validation)
- Multi-model switching (loads all joblib models in /models)
- SHAP + optional LIME hybrid explainability (sampled)
- Metrics & charts (ROC / PR / confusion matrix)
- In-app lightweight data cleaning UI
- Role-based auth (admin / viewer) via st.secrets['users']
- Dark mode toggle
- Admin prediction logs (JSONL)
- Retrain trigger (admin only)

Usage:
    streamlit run streamlit_app_improved.py

Notes:
- Place model files in models/ as joblib (e.g., models/lgbm_model.joblib).
- Place preprocessor at models/preprocessors/preprocessor.joblib.
- Place feature metadata at models/feature_metadata.json (with key 'feature_names').
- Configure secrets: .streamlit/secrets.toml with:
    [users]
    admin = "AdminPassword123"
    viewer = "ViewerPassword456"
"""

import os
import json
import joblib
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

# Optional explainability imports
import shap
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

# -------------------------
# Paths / config
# -------------------------
PROCESSED_CSV = "data/processed/default_of_credit_card_clients.csv"
MODELS_DIR = "models"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessors", "preprocessor.joblib")
FEATURE_META_PATH = os.path.join(MODELS_DIR, "feature_metadata.json")
LOGS_DIR = "logs"
PREDICTION_LOG = os.path.join(LOGS_DIR, "predictions.jsonl")

os.makedirs(LOGS_DIR, exist_ok=True)

# -------------------------
# Helpers: Loading artifacts
# -------------------------
@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, Any]:
    """Load all joblib models in models/ directory (files ending with .joblib)."""
    models = {}
    if not os.path.exists(MODELS_DIR):
        return models
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith(".joblib"):
            key = os.path.splitext(fname)[0]
            path = os.path.join(MODELS_DIR, fname)
            try:
                models[key] = joblib.load(path)
            except Exception as e:
                st.warning(f"Failed to load model {fname}: {e}")
    return models


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            return joblib.load(PREPROCESSOR_PATH)
        except Exception as e:
            st.error(f"Failed to load preprocessor: {e}")
    return None


@st.cache_data(show_spinner=False)
def load_processed_csv(path: str = PROCESSED_CSV) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_feature_meta() -> Optional[Dict[str, Any]]:
    if os.path.exists(FEATURE_META_PATH):
        try:
            with open(FEATURE_META_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to read feature metadata: {e}")
    return None


# -------------------------
# Auth & Roles
# -------------------------
def authenticate() -> str:
    """
    Role-based authentication. Expects st.secrets['users'] mapping username->password.
    Returns role: 'admin' or 'viewer' or '' if not authenticated.
    """
    st.sidebar.title("🔐 Login")
    users = {}
    try:
        users = st.secrets["users"]
    except Exception:
        # fallback: allow a default user for local dev (username=admin, password=admin)
        # but do not recommend in production
        users = {}

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if "role" not in st.session_state:
        st.session_state.role = ""

    if st.sidebar.button("Login"):
        if username in users and users[username] == password:
            role = "admin" if username == "admin" else "viewer"
            st.session_state.role = role
            st.success(f"Logged in as {role}")
        else:
            # local fallback for convenience if secrets not set
            if not users and username == "admin" and password == "admin":
                st.session_state.role = "admin"
                st.warning("Using local fallback admin/admin credentials (only for dev).")
            else:
                st.error("Invalid credentials. Configure `.streamlit/secrets.toml` with [users].")

    if st.session_state.role == "":
        st.stop()
    return st.session_state.role


# -------------------------
# UI Utilities
# -------------------------
def dark_mode_toggle():
    st.sidebar.markdown("### 🌓 Theme")
    mode = st.sidebar.radio("Select Theme", ["Light", "Dark"])
    css_dark = """
    <style>
        .reportview-container, .main, header, .stApp { background-color: #0e1117; color: #e6eef8; }
        .stButton>button { color: #e6eef8; }
    </style>
    """
    css_light = """
    <style>
        .reportview-container, .main, header, .stApp { background-color: #ffffff; color: #050505; }
    </style>
    """
    st.markdown(css_dark if mode == "Dark" else css_light, unsafe_allow_html=True)


def get_template_csv(df_example: pd.DataFrame) -> bytes:
    return df_example.head(20).to_csv(index=False).encode("utf-8")


def validate_uploaded_df(df: pd.DataFrame, model_features: List[str]) -> List[str]:
    errors = []
    if df is None or df.shape[0] == 0:
        errors.append("Uploaded file is empty.")
        return errors

    missing = [c for c in model_features if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    extra = [c for c in df.columns if c not in model_features]
    if extra:
        errors.append(f"Extra columns detected (they will be ignored): {extra}")

    # Simple dtype checks
    for col in model_features:
        if col in df.columns:
            if df[col].dtype == object and df[col].nunique() > 1000:
                errors.append(f"Column '{col}' looks like free text or has high cardinality.")
    return errors


def predict_proba_for_model(model, X_proc: np.ndarray) -> np.ndarray:
    """Unified prediction handler returning probabilities for positive class."""
    # scikit-learn style
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_proc)
            # If two columns, return positive class
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1]
            return probs.ravel()
        # LightGBM Booster
        if hasattr(model, "predict"):
            preds = model.predict(X_proc, num_iteration=getattr(model, "best_iteration", None))
            return np.array(preds).ravel()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    raise RuntimeError("Model cannot produce probabilities.")


def log_prediction(user_role: str, input_data: Dict[str, Any], prediction: Any, model_name: str):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_role": user_role,
        "model": model_name,
        "input": input_data,
        "prediction": prediction,
    }
    with open(PREDICTION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# -------------------------
# Manual Input UX (smart widgets + validation)
# -------------------------
def infer_column_type(series: pd.Series) -> str:
    """Heuristic to infer type: 'number', 'bool', 'categorical', 'text'."""
    uniq = series.dropna().unique()
    if len(uniq) == 0:
        return "number"
    if set(map(lambda x: str(x).strip(), map(str, uniq))).issubset({"0", "1", "True", "False", "true", "false"}):
        return "bool"
    # numeric?
    try:
        _ = series.dropna().astype(float)
        return "number"
    except Exception:
        pass
    if len(uniq) <= 20:
        return "categorical"
    return "text"


def render_manual_input_ui(
    feature_names: List[str],
    df_example: Optional[pd.DataFrame],
    model,
    preprocessor,
    user_role: str,
    selected_model_name: str,
):
    st.header("✍️ Single Record Prediction — Manual Input (smart widgets)")

    if df_example is None:
        # fallback: blank fields
        df_example = pd.DataFrame({c: [0] for c in feature_names})

    input_vals = {}
    cols = st.columns(2)
    for i, col in enumerate(feature_names):
        series = df_example.get(col, pd.Series([0]))
        coltype = infer_column_type(series if isinstance(series, pd.Series) else pd.Series(series))
        with cols[i % 2]:
            if coltype == "number":
                example_val = float(series.dropna().iloc[0]) if len(series.dropna()) > 0 else 0.0
                v = st.number_input(col, value=example_val, format="%f")
                input_vals[col] = v
            elif coltype == "bool":
                example_val = bool(int(str(series.dropna().iloc[0]).strip())) if len(series.dropna()) > 0 else False
                v = st.checkbox(col, value=example_val)
                input_vals[col] = int(bool(v))
            elif coltype == "categorical":
                options = sorted(map(str, pd.Series(series).dropna().unique()))
                default = options[0] if options else ""
                v = st.selectbox(col, options, index=0 if options else 0)
                input_vals[col] = v
            else:
                default = str(series.dropna().iloc[0]) if len(series.dropna()) > 0 else ""
                v = st.text_input(col, value=default)
                input_vals[col] = v

    if st.button("🔮 Predict (Manual)"):
        if model is None or preprocessor is None:
            st.error("Model or preprocessor missing.")
            return
        try:
            single_df = pd.DataFrame([input_vals])
            X_proc = preprocessor.transform(single_df)
            proba = predict_proba_for_model(model, X_proc)
            pred = int((proba > 0.5).astype(int)[0])
            st.metric("Predicted probability (pos class)", f"{float(proba[0]):.4f}")
            st.success(f"Predicted label: {pred}")

            # Log prediction
            log_prediction(user_role, input_vals, {"proba": float(proba[0]), "label": pred}, selected_model_name)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# -------------------------
# In-app data cleaning (lightweight)
# -------------------------
def cleaning_ui(df: pd.DataFrame) -> pd.DataFrame:
    st.header("🧹 Data Cleaning Tool (light)")

    if df is None:
        st.error("No dataset available for cleaning.")
        st.stop()
    df = df.copy()

    col = st.selectbox("Select column to clean", df.columns)
    st.write(df[col].head(10))

    if st.checkbox("Fill missing values"):
        method = st.selectbox("Fill method", ["mean", "median", "mode", "constant"])
        if method == "constant":
            val = st.text_input("Constant value")
            df[col] = df[col].fillna(val)
        elif method == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        elif method == "median":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mean())

    if st.checkbox("Convert to numeric (coerce)"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if st.checkbox("Convert to categorical"):
        df[col] = df[col].astype("category")

    if st.checkbox("Drop column"):
        if st.button("Confirm drop"):
            df = df.drop(columns=[col])
            st.success(f"Dropped column {col}")

    if st.button("Save cleaned sample (to /tmp/cleaned_sample.csv)"):
        tmp = "tmp/cleaned_sample.csv"
        os.makedirs("tmp", exist_ok=True)
        df.head(500).to_csv(tmp, index=False)
        st.success(f"Saved sample to {tmp}")
        with open(tmp, "rb") as f:
            st.download_button("Download cleaned sample", data=f, file_name="cleaned_sample.csv")

    st.write(df.head())
    return df


# -------------------------
# Explainability: SHAP + LIME hybrid (uses sample)
# -------------------------
def hybrid_explainability_ui(model, preprocessor, df_processed, feature_names: List[str]):
    st.header("🔍 Explainability — SHAP + LIME Hybrid (sampled)")

    if model is None or preprocessor is None or df_processed is None:
        st.error("Model, preprocessor or data missing.")
        return

    max_sample = min(2000, len(df_processed))
    sample_n = st.slider("Background sample size for explainer", min_value=100, max_value=max_sample, value=500)

    sample = df_processed.sample(n=min(sample_n, len(df_processed)), random_state=42)
    X = sample.drop(columns=["target"]) if "target" in sample.columns else sample
    X_proc = preprocessor.transform(X)

    st.info("Computing SHAP explainer (tree explainer used for tree models). This may take a moment.")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_proc)
        st.success("SHAP computed — showing summary plot")
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, features=X_proc, feature_names=feature_names, show=False)
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.error(f"SHAP failed: {e}")

    if LIME_AVAILABLE:
        st.markdown("### LIME local explanation (first sample row)")
        try:
            lime_exp = LimeTabularExplainer(
                X_proc,
                feature_names=feature_names,
                discretize_continuous=True,
            )
            exp = lime_exp.explain_instance(
                X_proc[0],
                lambda arr: predict_proba_for_model(model, arr.reshape(1, -1)),
                num_features=min(10, len(feature_names)),
            )
            st.write(exp.as_list())
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"LIME failed: {e}")
    else:
        st.info("LIME not installed. Install `lime` to enable local explanations.")


# -------------------------
# Main App
# -------------------------
def main():
    st.set_page_config(page_title="Loan Default — Model Explorer", layout="wide")
    role = authenticate()
    dark_mode_toggle()

    # Load artifacts
    models = load_models()
    preprocessor = load_preprocessor()
    feature_meta = load_feature_meta()
    df_processed = load_processed_csv()

    # Sidebar controls
    st.sidebar.header("Navigation")
    pages = [
        "Dataset Preview",
        "Examples Gallery & Upload",
        "Predict (CSV)",
        "Predict (Manual)",
        "Explainability (SHAP+LIME)",
        "Metrics & Charts",
        "Data Cleaning",
    ]
    if role == "admin":
        pages.append("Retrain & Admin")
    page = st.sidebar.radio("Go to", pages)

    # Multi-model selector (if multiple models present)
    if models:
        selected_model_name = st.sidebar.selectbox("Select model", sorted(list(models.keys())), index=0)
        selected_model = models[selected_model_name]
    else:
        selected_model_name = None
        selected_model = None
        st.sidebar.warning("No models found in models/*.joblib")

    # Page: Dataset Preview
    if page == "Dataset Preview":
        st.header("📁 Processed Dataset Preview")
        if df_processed is None:
            st.error("Processed CSV not found at data/processed. Run training pipeline first.")
        else:
            st.write(f"Shape: {df_processed.shape}")
            st.dataframe(df_processed.head(300))

    # Page: Examples Gallery & Upload
    if page == "Examples Gallery & Upload":
        st.header("🖼️ Examples Gallery & Upload Template")
        if df_processed is not None:
            st.write("Download a template CSV (first 20 rows):")
            st.download_button(
                "⬇ Download template CSV",
                data=get_template_csv(df_processed),
                file_name="template_sample.csv",
                mime="text/csv",
            )
            st.markdown("Preview example rows")
            st.dataframe(df_processed.head(50))
        else:
            st.info("No processed CSV available to generate template.")

    # Page: Predict (CSV)
    if page == "Predict (CSV)":
        st.header("📤 Batch Prediction — upload CSV")
        uploaded = st.file_uploader("Choose CSV", type=["csv"])
        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded)
                st.write("Uploaded shape:", df_upload.shape)
                model_features = feature_meta.get("feature_names", df_upload.columns.tolist()) if feature_meta else df_upload.columns.tolist()
                errors = validate_uploaded_df(df_upload, model_features)
                if errors:
                    for e in errors:
                        st.warning(e)
                    st.stop()
                if selected_model is None or preprocessor is None:
                    st.error("Model or preprocessor missing. Cannot predict.")
                else:
                    X_input = df_upload[model_features].copy()
                    X_proc = preprocessor.transform(X_input)
                    proba = predict_proba_for_model(selected_model, X_proc)
                    preds = (proba > 0.5).astype(int)
                    out = df_upload.copy()
                    out["pred_proba"] = proba
                    out["pred_label"] = preds
                    st.success("Predictions ready")
                    st.dataframe(out.head(100))
                    csv_out = out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", data=csv_out, file_name="predictions.csv", mime="text/csv")

                    # Log each prediction (admins may want this)
                    for idx, row in out.head(100).iterrows():
                        log_prediction(role, row[model_features].to_dict(), {"proba": float(row["pred_proba"]), "label": int(row["pred_label"])}, selected_model_name)
            except Exception as e:
                st.error(f"Failed to process uploaded CSV: {e}")

    # Page: Predict (Manual)
    if page == "Predict (Manual)":
        st.header("✍️ Single Prediction")
        if feature_meta:
            feature_names = feature_meta.get("feature_names", [])
        elif df_processed is not None:
            feature_names = [c for c in df_processed.columns if c != "target"]
        else:
            feature_names = []
        render_manual_input_ui(feature_names, df_processed.head(200) if df_processed is not None else None, selected_model, preprocessor, role, selected_model_name)

    # Page: Explainability
    if page == "Explainability (SHAP+LIME)":
        if selected_model is None:
            st.error("No model selected.")
        else:
            feature_names = feature_meta.get("feature_names", None) if feature_meta else None
            hybrid_explainability_ui(selected_model, preprocessor, df_processed, feature_names)

    # Page: Metrics & Charts
    if page == "Metrics & Charts":
        st.header("📈 Metrics & Charts")
        if selected_model is None or preprocessor is None or df_processed is None:
            st.error("Model/preprocessor/data missing.")
        else:
            if "target" not in df_processed.columns:
                st.error("Processed CSV missing 'target' column for metrics.")
            else:
                X_all = df_processed.drop(columns=["target"])
                y_all = df_processed["target"]
                X_proc = preprocessor.transform(X_all)
                proba = predict_proba_for_model(selected_model, X_proc)
                preds = (proba > 0.5).astype(int)

                auc = roc_auc_score(y_all, proba)
                pr_auc = average_precision_score(y_all, proba)
                acc = accuracy_score(y_all, preds)
                cm = confusion_matrix(y_all, preds)

                st.metric("ROC-AUC", f"{auc:.4f}")
                st.metric("PR-AUC", f"{pr_auc:.4f}")
                st.metric("Accuracy", f"{acc:.4f}")

                st.subheader("Confusion Matrix")
                st.write(cm)

                fig1, ax1 = plt.subplots()
                RocCurveDisplay.from_predictions(y_all, proba, ax=ax1)
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots()
                PrecisionRecallDisplay.from_predictions(y_all, proba, ax=ax2)
                st.pyplot(fig2)

    # Page: Data Cleaning
    if page == "Data Cleaning":
        cleaned = cleaning_ui(df_processed)
        st.write("Preview cleaned sample:")
        st.write(cleaned.head())

    # Page: Retrain & Admin (admin only)
    if page == "Retrain & Admin":
        if role != "admin":
            st.error("Admin-only page.")
        else:
            st.header("🔁 Retrain & Admin Controls")
            if st.button("Run training pipeline (python -m src.models.train)"):
                st.info("Starting training process... check terminal logs.")
                subprocess.Popen(["python", "-m", "src.models.train"])
                st.success("Training started (background process).")

            st.markdown("### 🔐 Prediction logs (last 200 entries)")
            if os.path.exists(PREDICTION_LOG):
                with open(PREDICTION_LOG, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-200:]
                st.text("".join(lines[-50:]))
                with open(PREDICTION_LOG, "rb") as f:
                    st.download_button("⬇ Download full logs", data=f, file_name="predictions.jsonl")
            else:
                st.info("No prediction logs yet.")

    st.sidebar.markdown("---")
    st.sidebar.write("Logged in as:", role)


if __name__ == "__main__":
    main()


import sys
st.write("Python Path:", sys.executable)