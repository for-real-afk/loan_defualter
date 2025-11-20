import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    confusion_matrix, accuracy_score
)
import lightgbm as lgb

from src.data.load_data import load_data
from src.features.build_features import build_feature_pipeline

# Directories
MODEL_DIR = "models"
PREPROCESSOR_DIR = os.path.join(MODEL_DIR, "preprocessors")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.joblib")
PREPROCESSOR_PATH = os.path.join(PREPROCESSOR_DIR, "preprocessor.joblib")
FEATURE_META_PATH = os.path.join(MODEL_DIR, "feature_metadata.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREPROCESSOR_DIR, exist_ok=True)


def train_lgbm_model(X_train, y_train, X_val, y_val):
    print("Starting LightGBM training...")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "num_leaves": 31,
        "seed": 42,
        "n_jobs": -1,
        "scale_pos_weight": len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        "verbose": -1
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    print(f"Best iteration: {model.best_iteration}")
    return model


def evaluate_model(model, X, y, name="Test"):
    y_proba = model.predict(X, num_iteration=model.best_iteration)
    y_pred = (y_proba > 0.5).astype(int)

    auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    print(f"\n--- {name} Evaluation ---")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {"auc": auc, "pr_auc": pr_auc}


def run_training_pipeline():
    print("--- Starting Training Pipeline ---")

    df = load_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"Dataset split → Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Preprocessing
    preprocessor = build_feature_pipeline(X_train, fit=True)

    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Train
    model = train_lgbm_model(X_train_proc, y_train, X_val_proc, y_val)

    # Evaluate
    evaluate_model(model, X_test_proc, y_test, "Holdout Test")

    # Save model and preprocessor
    joblib.dump(model, FINAL_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # Save metadata
    feature_meta = {"feature_names": list(X.columns)}
    json.dump(feature_meta, open(FEATURE_META_PATH, "w"), indent=4)

    print(f"Model saved: {FINAL_MODEL_PATH}")
    print(f"Preprocessor saved: {PREPROCESSOR_PATH}")
    print(f"Feature metadata saved: {FEATURE_META_PATH}")
    print("--- Pipeline Completed ---")


if __name__ == "__main__":
    run_training_pipeline()
