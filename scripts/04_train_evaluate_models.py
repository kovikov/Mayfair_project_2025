import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression # Keep baseline for comparison context
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Helper Functions (from baseline script) --- 
def calculate_iqr_bounds(data):
    """Calculates IQR bounds for numerical data."""
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    clr = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return metrics, clr, cm

def train_evaluate_log_models(processed_data_path, preprocessor_path, feature_names_path,
                                apply_iqr_filter=True, apply_smote=True, 
                                mlflow_experiment_name="Employee Churn Models",
                                random_state=42):
    """Loads data, preprocesses (IQR/SMOTE), trains, evaluates, and logs multiple models."""

    # --- 1. Load Processed Data --- 
    print("--- Loading processed data --- ")
    # (Error handling omitted for brevity, assuming files exist based on baseline success)
    data = np.load(processed_data_path)
    X_train_processed = data['X_train']
    y_train = data['y_train']
    X_test_processed = data['X_test']
    y_test = data['y_test']
    preprocessor = joblib.load(preprocessor_path)
    with open(feature_names_path, 'r') as f:
        processed_feature_names = [line.strip() for line in f.readlines()]
    print(f"Loaded data, preprocessor, and {len(processed_feature_names)} feature names.")
    # Adjust feature names if length mismatch (as in baseline)
    if X_train_processed.shape[1] != len(processed_feature_names):
             print(f"Warning: Adjusting feature names list length.")
             processed_feature_names = processed_feature_names[:X_train_processed.shape[1]]


    # --- 2. Apply IQR Filtering (Optional) --- 
    if apply_iqr_filter:
        print("\n--- Applying IQR Filtering (on training data only) ---")
        lower_bound, upper_bound = calculate_iqr_bounds(X_train_processed)
        outlier_mask = ((X_train_processed < lower_bound) | (X_train_processed > upper_bound)).any(axis=1)
        num_outliers = np.sum(outlier_mask)
        print(f"Identified {num_outliers} outlier rows.")
        X_train_filtered = X_train_processed[~outlier_mask]
        y_train_filtered = y_train[~outlier_mask]
        print(f"Filtered training data shape: X={X_train_filtered.shape}, y={y_train_filtered.shape}")
    else:
        print("\n--- Skipping IQR Filtering --- ")
        X_train_filtered = X_train_processed
        y_train_filtered = y_train

    # --- 3. Apply SMOTE (Optional) ---
    print(f"\n--- Initial Training Data Class Distribution ---")
    print(pd.Series(y_train_filtered).value_counts(normalize=True))
    if apply_smote:
        print("\n--- Applying SMOTE --- ")
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_filtered, y_train_filtered)
        print(f"Resampled training data shape: X={X_train_resampled.shape}, y={y_train_resampled.shape}")
        print(f"\n--- Resampled Training Data Class Distribution ---")
        print(pd.Series(y_train_resampled).value_counts(normalize=True))
    else:
        print("\n--- Skipping SMOTE --- ")
        X_train_resampled = X_train_filtered
        y_train_resampled = y_train_filtered

    # --- 4. Define Models --- 
    models = {
        "LogisticRegression": LogisticRegression(random_state=random_state, max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=random_state, n_estimators=100), # Default n_estimators
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state) 
    }

    # --- 5. MLflow Setup & Training Loop --- 
    print(f"\n--- Starting MLflow Runs for Experiment: '{mlflow_experiment_name}' ---")
    mlflow.set_experiment(mlflow_experiment_name)
    
    for model_name, model in models.items():
        print(f"\n===== Training {model_name} =====")
        with mlflow.start_run(run_name=model_name) as run: # Use model name for run name
            mlflow_run_id = run.info.run_id
            print(f"MLflow Run ID: {mlflow_run_id}")
            
            # Log data handling params
            mlflow.log_param("apply_iqr_filter", apply_iqr_filter)
            mlflow.log_param("apply_smote", apply_smote)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("model_type", model_name)
            
            # Log model-specific parameters (can be expanded)
            mlflow.log_params(model.get_params())

            # Train
            model.fit(X_train_resampled, y_train_resampled)
            print(f"{model_name} trained.")

            # Evaluate
            print(f"\n--- Evaluating {model_name} on Test Set ---")
            metrics, clr, cm = evaluate_model(model, X_test_processed, y_test)
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            print("\nClassification Report:")
            print(clr)
            print("Confusion Matrix:")
            print(cm)

            # Log metrics & artifacts
            print("\n--- Logging to MLflow --- ")
            mlflow.log_metrics(metrics)
            
            # Log reports/matrices as artifacts
            with open(f"{model_name}_classification_report.txt", "w") as f:
                f.write(clr)
            mlflow.log_artifact(f"{model_name}_classification_report.txt")
            os.remove(f"{model_name}_classification_report.txt")
            
            with open(f"{model_name}_confusion_matrix.txt", "w") as f:
                 f.write(np.array2string(cm))
            mlflow.log_artifact(f"{model_name}_confusion_matrix.txt")
            os.remove(f"{model_name}_confusion_matrix.txt")

            mlflow.sklearn.log_model(model, model_name) # Log model with its name
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
            mlflow.log_artifact(feature_names_path, artifact_path="features")
            print(f"Logged results and artifacts for {model_name}.")
            
    print("\n--- Model Training & Evaluation Script finished successfully! --- ")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    processed_data_file = os.path.join(base_dir, '..', 'data', 'processed_data.npz')
    preprocessor_file = os.path.join(base_dir, '..', 'models', 'preprocessor.joblib')
    feature_names_file = os.path.join(base_dir, '..', 'data', 'processed_feature_names.txt')

    # Check if required files exist before running
    if not all(os.path.exists(f) for f in [processed_data_file, preprocessor_file, feature_names_file]):
        print("Error: One or more required files (processed data, preprocessor, feature names) not found.")
        print("Please run the preprocessing script (02_preprocessing.py) first.")
    else:
        train_evaluate_log_models(
            processed_data_path=processed_data_file, 
            preprocessor_path=preprocessor_file,
            feature_names_path=feature_names_file,
            apply_iqr_filter=True, # Consistent with baseline run
            apply_smote=True      # Consistent with baseline run
        ) 