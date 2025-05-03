from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import numpy as np

app = FastAPI(title="Employee Churn Prediction API", version="1.0.0")

# --- Load Model Artifacts --- 
model = None
preprocessor = None
feature_names_in = None # Original feature names expected by preprocessor

# Determine absolute paths relative to this script file
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "models", "final_churn_model.joblib")
preprocessor_path = os.path.join(base_dir, "..", "models", "preprocessor.joblib")
# We might need original feature names if preprocessor doesn't store them
# Let's try loading preprocessor first, it might contain them

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
    elif hasattr(preprocessor, 'transformers_'): 
         # Older sklearn might store in transformers_
         feature_names_in = []
         for name, trans, cols in preprocessor.transformers_:
             if trans != 'drop' and cols: # Check if cols is not empty
                feature_names_in.extend(cols)
         # Handle remainder='passthrough' - need original column names
         # This part is tricky without knowing the exact structure/original columns
         # We might need to load original columns from the data or saved list
         print("Warning: Attempting to infer input features from transformers. Order might be incorrect.")
    else:
        print("Warning: Could not automatically determine input feature names from preprocessor.")
        # Define expected input columns manually as a fallback 
        # **MUST match the order and names used when fitting the preprocessor**
        feature_names_in = [
            'SATISFACTION_LEVEL', 'LAST_EVALUATION', 'NUMBER_PROJECT', 
            'AVERAGE_MONTLY_HOURS', 'TIME_SPEND_COMPANY', 'WORK_ACCIDENT', 
            'PROMOTION_LAST_5YEARS', 'DEPARTMENTS', 'SALARY', 'ABSENTEEISM', 
            'MANAGER_FEEDBACK_SCORE', 'REMOTE_WORK', 'ENGAGEMENT_SCORE' 
            # Excludes EMPLOYEE_ID (dropped) and QUIT_THE_COMPANY (target)
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
    DEPARTMENTS: str  # Categorical
    SALARY: str      # Ordinal Categorical
    ABSENTEEISM: int
    JOB_ROLE: str    # Categorical 
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

@app.post("/predict", response_model=PredictionOut)
def predict_churn(features: EmployeeFeatures):
    """
    Predicts employee churn based on input features.
    Requires all features used during model training.
    
    Returns prediction (0=stay, 1=quit) and probability of quitting.
    """
    global model, preprocessor, feature_names_in # Allow modification if needed later
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or Preprocessor not loaded. API unavailable.")

    try:
        # Convert input features to DataFrame format expected by the preprocessor
        # Ensure the order matches the feature_names_in obtained or defined earlier
        if feature_names_in is None:
             raise HTTPException(status_code=500, detail="Input feature names for preprocessor are missing.")
             
        feature_dict = features.dict()
        # Create DataFrame with columns in the correct order
        input_df = pd.DataFrame([feature_dict])
        # Reorder columns to match the order used during preprocessor fitting
        input_df = input_df[feature_names_in] 
        
        # Apply the loaded preprocessor
        processed_features = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(processed_features)[0]
        
        # Get probability (probability of the positive class, i.e., quitting)
        probability = model.predict_proba(processed_features)[0][1] 
        
        return {"prediction": int(prediction), "probability_quit": float(probability)}
        
    except KeyError as e:
        # Handle case where input data is missing an expected column
        raise HTTPException(status_code=422, detail=f"Missing feature in input data: {e}")
    except ValueError as e:
        # Handle cases like invalid values that cause errors during preprocessing/prediction
         raise HTTPException(status_code=422, detail=f"Invalid value or type in input data: {e}")
    except Exception as e:
        # Catch-all for other unexpected errors
        print(f"Error during prediction processing: {e}") # Log the error server-side
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

@app.get("/health")
def health_check():
    """Basic health check for the API."""
    preprocessor_status = "OK" if preprocessor is not None else "Preprocessor not loaded"
    model_status = "OK" if model is not None else "Model not loaded"
    
    if preprocessor is not None and model is not None:
        status_code = 200
        overall_status = "OK"
    else:
        status_code = 503 # Service Unavailable
        overall_status = "Unavailable"
        
    return {
        "status": overall_status,
        "preprocessor_status": preprocessor_status,
        "model_status": model_status,
        # Consider adding checks for feature_names_in availability too
        # "feature_names_status": "OK" if feature_names_in else "Not Found"
    }# Returning status code is implicitly handled by response framework

# To run the app (from the project root directory):
# uvicorn src.main:app --reload 