import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

# Define paths for artifacts
PREPROCESSOR_DIR = 'models/preprocessors'
PREPROCESSOR_PATH = os.path.join(PREPROCESSOR_DIR, 'preprocessor.joblib')

def build_feature_pipeline(X: pd.DataFrame, fit: bool = True) -> ColumnTransformer:
    """
    Builds, fits, and/or saves the preprocessing ColumnTransformer pipeline.

    The pipeline handles:
    1. Numerical features: Imputation (median) and Scaling (StandardScaler).
    2. Categorical features: Imputation (most frequent) and One-Hot Encoding.
    
    Args:
        X (pd.DataFrame): The features to process.
        fit (bool): If True, fits the pipeline and saves it. If False, loads the saved pipeline.

    Returns:
        ColumnTransformer: The fitted or loaded preprocessing pipeline.
    """
    # Identify feature types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Manually define known categorical features from the UCI dataset
    manual_cats = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    
    # Separate purely numerical and categorical (that might be int/float)
    cat_cols = [c for c in manual_cats if c in X.columns]
    numeric_cols = [c for c in numeric_cols if c not in cat_cols]
    
    print(f"Numerical columns ({len(numeric_cols)}): {numeric_cols[:3]}...")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    # 1. Numerical Pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Use sparse_output=False for dense matrix
    ])

    # 3. Combined Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, cat_cols)
        ],
        remainder='passthrough', # Keep any unlisted columns (e.g., if we missed some)
        verbose_feature_names_out=False # Use simple names
    )
    preprocessor.set_output(transform="pandas") # Output DataFrame with column names

    if fit:
        print("Fitting preprocessing pipeline...")
        # Fit and transform
        preprocessor.fit(X)
        
        # Save preprocessor
        os.makedirs(PREPROCESSOR_DIR, exist_ok=True)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        print(f"Saved preprocessor to {PREPROCESSOR_PATH}")
        return preprocessor
    else:
        # Load preprocessor
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print(f"Loaded preprocessor from {PREPROCESSOR_PATH}")
            return preprocessor
        except FileNotFoundError:
            raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}. Run training first.")