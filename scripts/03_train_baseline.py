import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore')

def calculate_iqr_bounds(data):
    """Calculates IQR bounds for numerical data."""
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def train_baseline_model(processed_data_path, preprocessor_path, feature_names_path,
                           apply_iqr_filter=True, apply_smote=True, 
                           mlflow_experiment_name="Employee Churn Baseline",
                           random_state=42):
    """Loads processed data, applies optional IQR/SMOTE, trains baseline, logs to MLflow."""
    
    # --- 1. Load Processed Data --- 
    print("--- Loading processed data --- ")
    try:
        data = np.load(processed_data_path)
        X_train_processed = data['X_train']
        y_train = data['y_train']
        X_test_processed = data['X_test']
        y_test = data['y_test']
        print(f"Loaded data: Train X={X_train_processed.shape}, Test X={X_test_processed.shape}")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {processed_data_path}.")
        return
    except KeyError as e:
        print(f"Error: Missing key {e} in processed data file.")
        return

    try:
        preprocessor = joblib.load(preprocessor_path)
        print(f"Loaded preprocessor from {preprocessor_path}")
    except FileNotFoundError:
        print(f"Error: Preprocessor file not found at {preprocessor_path}.")
        return
        
    try:
        with open(feature_names_path, 'r') as f:
            processed_feature_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(processed_feature_names)} feature names from {feature_names_path}")
        # Sanity check length
        if X_train_processed.shape[1] != len(processed_feature_names):
             print(f"Warning: Mismatch between processed data columns ({X_train_processed.shape[1]}) and feature names ({len(processed_feature_names)}).")
             # Attempt to trim/adjust names if possible, otherwise proceed with caution
             processed_feature_names = processed_feature_names[:X_train_processed.shape[1]]
             print(f"Adjusted feature names list length to {len(processed_feature_names)}")

    except FileNotFoundError:
        print(f"Error: Feature names file not found at {feature_names_path}.")
        processed_feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])] # Fallback names
        print(f"Using fallback feature names.")


    # --- 2. IQR Filtering (Optional) --- 
    if apply_iqr_filter:
        print("\n--- Applying IQR Filtering (on training data only) ---")
        lower_bound, upper_bound = calculate_iqr_bounds(X_train_processed)
        
        # Identify outliers column by column
        outlier_mask = ((X_train_processed < lower_bound) | (X_train_processed > upper_bound)).any(axis=1)
        
        num_outliers = np.sum(outlier_mask)
        print(f"Identified {num_outliers} rows with outliers based on IQR across all features.")
        
        X_train_filtered = X_train_processed[~outlier_mask]
        y_train_filtered = y_train[~outlier_mask]
        print(f"Filtered training data shape: X={X_train_filtered.shape}, y={y_train_filtered.shape}")
    else:
        print("\n--- Skipping IQR Filtering --- ")
        X_train_filtered = X_train_processed
        y_train_filtered = y_train

    # --- 3. Handle Class Imbalance using SMOTE (Optional) ---
    print(f"\n--- Initial Training Data Class Distribution ---")
    print(pd.Series(y_train_filtered).value_counts(normalize=True))
    
    if apply_smote:
        print("\n--- Applying SMOTE to handle class imbalance (on filtered training data) ---")
        smote = SMOTE(random_state=random_state)
        try:
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_filtered, y_train_filtered)
            print(f"Resampled training data shape: X={X_train_resampled.shape}, y={y_train_resampled.shape}")
            print(f"\n--- Resampled Training Data Class Distribution ---")
            print(pd.Series(y_train_resampled).value_counts(normalize=True))
        except Exception as e:
            print(f"Error during SMOTE: {e}. Proceeding without resampling.")
            apply_smote = False # Disable flag if SMOTE failed
            X_train_resampled = X_train_filtered
            y_train_resampled = y_train_filtered
            
    else:
        print("\n--- Skipping SMOTE --- ")
        X_train_resampled = X_train_filtered
        y_train_resampled = y_train_filtered

    # --- 4. MLflow Setup --- 
    print(f"\n--- Starting MLflow Run for Experiment: '{mlflow_experiment_name}' ---")
    mlflow.set_experiment(mlflow_experiment_name)
    
    with mlflow.start_run() as run:
        mlflow_run_id = run.info.run_id
        print(f"MLflow Run ID: {mlflow_run_id}")
        
        mlflow.log_param("apply_iqr_filter", apply_iqr_filter)
        mlflow.log_param("apply_smote", apply_smote)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("baseline_model", "LogisticRegression")

        # --- 5. Train Baseline Model --- 
        print("\n--- Training Baseline Model (Logistic Regression) --- ")
        baseline_model = LogisticRegression(random_state=random_state, max_iter=1000) # Increase max_iter for convergence
        baseline_model.fit(X_train_resampled, y_train_resampled)
        print("Baseline model trained.")

        # --- 6. Evaluation --- 
        print("\n--- Evaluating Baseline Model on Test Set ---")
        y_pred = baseline_model.predict(X_test_processed)
        y_proba = baseline_model.predict_proba(X_test_processed)[:, 1] # Probability of positive class
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        print("\nClassification Report:")
        clr = classification_report(y_test, y_pred)
        print(clr)
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # --- 7. MLflow Logging --- 
        print("\n--- Logging to MLflow --- ")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log classification report as text artifact
        with open("classification_report.txt", "w") as f:
            f.write(clr)
        mlflow.log_artifact("classification_report.txt")
        os.remove("classification_report.txt") # Clean up temp file
        
        # Log confusion matrix (e.g., as text or image)
        # For simplicity, logging as text here
        with open("confusion_matrix.txt", "w") as f:
             f.write(np.array2string(cm))
        mlflow.log_artifact("confusion_matrix.txt")
        os.remove("confusion_matrix.txt")

        # Log the trained model
        mlflow.sklearn.log_model(baseline_model, "baseline_model")
        print("Logged baseline model to MLflow.")
        
        # Log the preprocessor as an artifact
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
        print(f"Logged preprocessor from {preprocessor_path} to MLflow.")

        # Log feature names as an artifact
        mlflow.log_artifact(feature_names_path, artifact_path="features")
        print(f"Logged feature names from {feature_names_path} to MLflow.")

        # --- 8. Save Model Explicitly (Optional) --- 
        # model_save_path = os.path.join(os.path.dirname(__file__), '../models', 'baseline_model.joblib')
        # joblib.dump(baseline_model, model_save_path)
        # print(f"Saved baseline model explicitly to {model_save_path}")

    print(f"\n--- MLflow Run ({mlflow_run_id}) completed. --- ")
    print(f"--- Baseline Training Script finished successfully! --- ")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    processed_data_file = os.path.join(base_dir, '..', 'data', 'processed_data.npz')
    preprocessor_file = os.path.join(base_dir, '..', 'models', 'preprocessor.joblib')
    feature_names_file = os.path.join(base_dir, '..', 'data', 'processed_feature_names.txt')

    # Check if required files exist before running
    if not os.path.exists(processed_data_file):
        print(f"Error: Processed data file not found: {processed_data_file}")
        print("Please run the preprocessing script (02_preprocessing.py) first.")
    elif not os.path.exists(preprocessor_file):
        print(f"Error: Preprocessor file not found: {preprocessor_file}")
        print("Please run the preprocessing script (02_preprocessing.py) first.")
    elif not os.path.exists(feature_names_file):
        print(f"Error: Feature names file not found: {feature_names_file}")
        print("Please run the preprocessing script (02_preprocessing.py) first.")
    else:
        train_baseline_model(
            processed_data_path=processed_data_file, 
            preprocessor_path=preprocessor_file,
            feature_names_path=feature_names_file,
            apply_iqr_filter=True, # Set to False to disable IQR filtering
            apply_smote=True      # Set to False to disable SMOTE
        ) 