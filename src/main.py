from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import numpy as np

# --- Environment Variable for API Key --- 
# IMPORTANT: Set this environment variable in your deployment environment (e.g., Render)
# Use a strong, randomly generated key in production.
EXPECTED_API_KEY = os.getenv("API_KEY", "default_insecure_key_for_local_dev") # Fallback for local dev ONLY
if EXPECTED_API_KEY == "default_insecure_key_for_local_dev":
    print("\n**************************************************************************")
    print("WARNING: Using default insecure API key. Set the API_KEY environment variable in production!")
    print("**************************************************************************\n")

# --- API Key Security Setup --- 
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(key: str = Security(api_key_header)):
    """Dependency function to validate the API key."""
    if key == EXPECTED_API_KEY:
        return key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

# --- FastAPI App Initialization ---
app = FastAPI(title="Employee Churn Prediction API", version="1.0.0")

# --- Load Model Artifacts --- 
model = None
preprocessor = None
feature_names_in = None # Original feature names expected by preprocessor

# Determine absolute paths relative to this script file
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "models", "final_churn_model.joblib")
preprocessor_path = os.path.join(base_dir, "..", "models", "preprocessor.joblib")

print(f"Attempting to load model from: {model_path}")
print(f"Attempting to load preprocessor from: {preprocessor_path}")

try:
    model = joblib.load(model_path)
    print("Successfully loaded final_churn_model.joblib")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    # Keep model as None, health check will fail
except Exception as e:
    print(f"Error loading model: {e}")

try:
    preprocessor = joblib.load(preprocessor_path)
    print("Successfully loaded preprocessor.joblib")
    # Try to get input features names expected by the preprocessor
    # This depends on the scikit-learn version and how the ColumnTransformer was fitted.
    if hasattr(preprocessor, 'feature_names_in_'):
        feature_names_in = preprocessor.feature_names_in_
        print(f"Successfully loaded input feature names from preprocessor: {list(feature_names_in)}")
    elif hasattr(preprocessor, 'transformers_'): 
         # Older sklearn might store in transformers_
         feature_names_in = []
         # Keep track of columns used by transformers to infer remainder columns
         used_columns = set()
         for name, trans, cols in preprocessor.transformers_:
             if trans != 'drop' and cols:
                feature_names_in.extend(cols)
                used_columns.update(cols)
         
         # Attempt to infer remainder columns (this requires knowing original columns)
         # This is brittle; explicitly saving/loading original columns is better.
         # Assuming 'passthrough' keeps remaining original columns IN THEIR ORIGINAL ORDER
         # We need a reliable way to get the original columns that X had when preprocessor was fit.
         # Loading original data or a saved column list is the robust way.
         # FALLBACK (less reliable): Manually define original expected cols (excluding target)
         original_cols_fallback = [
            'SATISFACTION_LEVEL', 'LAST_EVALUATION', 'NUMBER_PROJECT', 
            'AVERAGE_MONTLY_HOURS', 'TIME_SPEND_COMPANY', 'WORK_ACCIDENT', 
            'PROMOTION_LAST_5YEARS', 'DEPARTMENTS', 'SALARY', 'ABSENTEEISM', 
            'MANAGER_FEEDBACK_SCORE', 'JOB_ROLE', 'REMOTE_WORK', 'ENGAGEMENT_SCORE'
         ] # Based on previous exploration, verify this list!
         remainder_cols = [col for col in original_cols_fallback if col not in used_columns]
         feature_names_in.extend(remainder_cols) # Add inferred remainder columns
         print(f"Warning: Inferred input features from transformers. Order relies on manual list: {feature_names_in}")
    else:
        print("Warning: Could not automatically determine input feature names from preprocessor.")
        # Define expected input columns manually as a fallback 
        # **MUST match the order and names used when fitting the preprocessor**
        feature_names_in = [
            'SATISFACTION_LEVEL', 'LAST_EVALUATION', 'NUMBER_PROJECT', 
            'AVERAGE_MONTLY_HOURS', 'TIME_SPEND_COMPANY', 'WORK_ACCIDENT', 
            'PROMOTION_LAST_5YEARS', 'DEPARTMENTS', 'SALARY', 'ABSENTEEISM', 
            'MANAGER_FEEDBACK_SCORE', 'JOB_ROLE', 'REMOTE_WORK', 'ENGAGEMENT_SCORE'
        ]
        print(f"Using fallback feature names: {feature_names_in}")

except FileNotFoundError:
    print(f"Error: Preprocessor file not found at {preprocessor_path}")
    # Keep preprocessor as None, health check will fail
except Exception as e:
    print(f"Error loading preprocessor: {e}")

# --- API Data Models --- 

# Update Pydantic model to include all expected raw features
class EmployeeFeatures(BaseModel):
    SATISFACTION_LEVEL: float
    LAST_EVALUATION: float
    NUMBER_PROJECT: int
    AVERAGE_MONTLY_HOURS: int
    TIME_SPEND_COMPANY: int
    WORK_ACCIDENT: int
    PROMOTION_LAST_5YEARS: int
    DEPARTMENTS: str
    SALARY: str
    ABSENTEEISM: int
    JOB_ROLE: str
    MANAGER_FEEDBACK_SCORE: float
    REMOTE_WORK: int
    ENGAGEMENT_SCORE: float
    # Ensure names match columns used during preprocessor fitting (case-sensitive)

class PredictionOut(BaseModel):
    prediction: int # 0 for stay, 1 for quit
    probability_quit: float | None = None # Probability of class 1 (quit)

# --- API Endpoints --- 

@app.get("/")
def read_root():
    return {"message": "Welcome to the Employee Churn Prediction API. Use POST /predict for predictions."}

# Apply the API key dependency to the predict endpoint
@app.post("/predict", response_model=PredictionOut, dependencies=[Depends(get_api_key)])
def predict_churn(features: EmployeeFeatures):
    """
    Predicts employee churn based on input features.
    Requires all features used during model training.
    Requires valid API key in 'X-API-Key' header.
    
    Returns prediction (0=stay, 1=quit) and probability of quitting.
    """
    global model, preprocessor, feature_names_in # Allow modification if needed later
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or Preprocessor not loaded. API unavailable.")

    try:
        # Convert input features to DataFrame format expected by the preprocessor
        if feature_names_in is None:
             # This condition should ideally not be reached if startup loading is robust
             raise HTTPException(status_code=500, detail="Input feature names for preprocessor are missing.")
             
        feature_dict = features.dict()
        # Create DataFrame with columns in the correct order defined by feature_names_in
        try:
            input_df = pd.DataFrame([feature_dict])
            # Reorder columns to match the exact order preprocessor was trained on
            input_df = input_df[feature_names_in] 
        except KeyError as e:
             raise HTTPException(status_code=422, detail=f"Input data missing expected feature column: {e}. Expected columns: {feature_names_in}")
        
        # Apply the loaded preprocessor
        processed_features = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(processed_features)[0]
        
        # Get probability (probability of the positive class, i.e., quitting)
        probability = model.predict_proba(processed_features)[0][1] 
        
        return {"prediction": int(prediction), "probability_quit": float(probability)}
        
    except ValueError as e:
        # Handle cases like invalid values that cause errors during preprocessing/prediction
         raise HTTPException(status_code=422, detail=f"Invalid value or type in input data during preprocessing: {e}")
    except Exception as e:
        # Catch-all for other unexpected errors during prediction
        print(f"Error during prediction processing: {e}") # Log the error server-side
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

@app.get("/health")
def health_check():
    """Basic health check for the API."""
    preprocessor_status = "OK" if preprocessor is not None else "Preprocessor not loaded"
    model_status = "OK" if model is not None else "Model not loaded"
    feature_names_status = "OK" if feature_names_in is not None else "Not Found"
    
    if preprocessor is not None and model is not None and feature_names_in is not None:
        overall_status = "OK"
    else:
        overall_status = "Unavailable"
        
    return {
        "status": overall_status,
        "preprocessor_status": preprocessor_status,
        "model_status": model_status,
        "feature_names_status": feature_names_status
    }

# To run the app (from the project root directory):
# uvicorn src.main:app --reload 