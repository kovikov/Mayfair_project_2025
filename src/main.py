from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Employee Churn Prediction API", version="0.1.0")

# Placeholder: Load the trained model (replace with actual model loading, possibly via MLflow)
# model_path = os.path.join("models", "best_model.joblib") 
# try:
#     model = joblib.load(model_path)
# except FileNotFoundError:
#     model = None # Or load a default/dummy model

model = None # Placeholder

# Define the input data model using Pydantic
# Adjust features based on the final selected features after preprocessing
class EmployeeFeatures(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    number_project: int
    average_montly_hours: int
    time_spend_company: int
    work_accident: int
    promotion_last_5years: int
    # Add other features as needed, ensure types match
    # e.g., departments: str, salary: str, etc.
    # Example:
    # departments: str
    # salary: str

class PredictionOut(BaseModel):
    prediction: int # 0 for stay, 1 for quit (adjust as needed)
    probability: float | None = None # Optional: probability of quitting

@app.get("/")
def read_root():
    return {"message": "Welcome to the Employee Churn Prediction API. Use /predict endpoint for predictions."}

@app.post("/predict", response_model=PredictionOut)
def predict_churn(features: EmployeeFeatures):
    """
    Predicts employee churn based on input features.

    - **satisfaction_level**: Employee satisfaction level (0-1)
    - **last_evaluation**: Score of last evaluation (0-1)
    - **number_project**: Number of projects assigned to
    - **average_montly_hours**: Average monthly hours worked
    - **time_spend_company**: Years spent with the company
    - **work_accident**: Whether the employee had a work accident (1=yes, 0=no)
    - **promotion_last_5years**: Whether the employee was promoted in the last 5 years (1=yes, 0=no)
    
    *(Add descriptions for other features here)*
    
    Returns prediction (0 or 1) and optionally the probability.
    """
    if model is None:
        return {"prediction": -1, "probability": None} # Indicate model not loaded

    # Convert input features to DataFrame format expected by the model
    # This needs adjustment based on actual preprocessing steps (scaling, encoding)
    feature_dict = features.dict()
    # Placeholder: ensure feature names and order match training data
    # May need one-hot encoding for categorical features, scaling for numerical
    try:
        # features_df = pd.DataFrame([feature_dict]) 
        # Placeholder: Apply the same preprocessing as during training
        # prediction = model.predict(features_df)[0]
        # probability = model.predict_proba(features_df)[0][1] # Prob of class 1 (quit)
        
        # --- Placeholder Response --- 
        # Replace with actual prediction logic 
        prediction_value = 0 if features.satisfaction_level > 0.5 else 1 
        probability_value = 1.0 - features.satisfaction_level 
        # --- End Placeholder --- 
        
        return {"prediction": prediction_value, "probability": probability_value}
    except Exception as e:
        # Handle potential errors during prediction (e.g., invalid input format)
        # Log the error
        print(f"Error during prediction: {e}")
        # Return a default or error indication
        # Consider using FastAPI's HTTPException for proper error responses
        return {"prediction": -2, "probability": None} 


# Add health check endpoint (optional but good practice)
@app.get("/health")
def health_check():
    # Basic health check: Check if model is loaded
    status = "OK" if model is not None else "Model not loaded"
    return {"status": status}

# To run the app (from the project root directory):
# uvicorn src.main:app --reload 