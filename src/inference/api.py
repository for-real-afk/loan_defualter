from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import shap
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Artifact Loading ---
# Define paths for artifacts (relative to the project root, assuming execution starts from there)
MODEL_PATH = 'models/final_model.joblib'
PREPROCESSOR_PATH = 'models/preprocessors/preprocessor.joblib'

# Check if artifacts exist before attempting to load
if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    logging.error("Model or preprocessor artifact not found. Ensure you run src/models/train.py first.")
    # In a non-Docker environment, we still raise the error to prevent API from starting without files
    raise FileNotFoundError("Missing model artifacts. Cannot start API.")

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logging.info("Model and Preprocessor loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load artifacts: {e}")
    raise

# Initialize SHAP Explainer (Tree Explainer is efficient for LightGBM)
try:
    # Initializing TreeExplainer for LightGBM model
    explainer = shap.TreeExplainer(model)
    logging.info("SHAP Explainer initialized.")
except Exception as e:
    logging.warning(f"Failed to initialize SHAP Explainer: {e}")
    explainer = None


# --- FastAPI Setup and Data Validation ---
app = FastAPI(
    title="Loan Default Predictor API",
    description="A service for predicting credit card default using the UCI dataset model."
)

# Pydantic model for input data validation
class InputData(BaseModel):
    # This structure must match the expected raw input columns *exactly*
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

# --- Utility Function for SHAP (Placeholder) ---
def get_shap_explanation(processed_data: pd.DataFrame):
    """Generates a simplified SHAP explanation (top 3 features)."""
    if not explainer:
        return "SHAP Explainer is not available."
    
    # Calculate SHAP values for the single instance
    # Note: Use the underlying numpy array for SHAP calculation for best compatibility
    # [1] extracts the SHAP values for the positive class (default)
    shap_values = explainer.shap_values(processed_data.iloc[0].to_numpy())[1] 
    
    # Get the feature names after preprocessing
    feature_names = processed_data.columns.tolist()
    
    # Combine feature names and SHAP values
    features_and_shap = pd.Series(shap_values, index=feature_names)
    
    # Get top 3 features by absolute SHAP value
    top_3_features = features_and_shap.abs().nlargest(3).index.tolist()
    
    # Calculate the base value (expected value)
    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    
    explanation = {
        'base_value': float(expected_value),
        'top_features': {
            feat: float(features_and_shap[feat])
            for feat in top_3_features
        }
    }
    return explanation


# --- API Endpoints ---

@app.get('/health')
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "model_ready": True}

@app.post('/predict')
def predict(payload: InputData):
    """
    Accepts raw feature data, preprocesses it, and returns the prediction
    probability and a simplified SHAP explanation.
    """
    try:
        start_time = pd.Timestamp.now()
        
        # 1. Convert payload to DataFrame (single row)
        raw_df = pd.DataFrame([payload.model_dump()])
        
        # 2. Preprocess
        # Preprocessor returns a DataFrame with feature names (thanks to set_output(transform="pandas") in build_features.py)
        processed_df = preprocessor.transform(raw_df)
        
        # 3. Predict Probability
        # model.predict() returns a numpy array of probabilities for class 1
        probability = model.predict(processed_df, raw_score=False)[0] 
        
        # 4. Determine final prediction label
        prediction_label = 1 if probability > 0.5 else 0
        
        # 5. Generate SHAP Explanation
        shap_explanation = get_shap_explanation(processed_df)

        latency = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Log the prediction request and response
        logging.info(f"Prediction made (Latency: {latency:.4f}s): P={probability:.4f}, Label={prediction_label}")

        return {
            'probability_of_default': float(probability),
            'prediction_label': prediction_label,
            'explanation_summary': "Probability > 0.5 is default.",
            'shap_values': shap_explanation,
            'latency_s': latency
        }

    except Exception as e:
        logging.error(f"Prediction failed due to error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error during prediction: {str(e)}")