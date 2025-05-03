import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix, make_scorer
)
from imblearn.over_sampling import SMOTE
import joblib
import warnings
from scipy.stats import randint

warnings.filterwarnings('ignore')

# --- Helper Functions (from previous scripts) --- 
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

def tune_log_final_model(processed_data_path, preprocessor_path, feature_names_path,
                         apply_iqr_filter=True, apply_smote=True, 
                         mlflow_experiment_name="Employee Churn Models",
                         n_iter_search=20, # Number of parameter settings that are sampled
                         cv_folds=5, # Number of cross-validation folds
                         random_state=42):
    """Loads data, preprocesses, tunes RF, evaluates, and logs the best model."""

    # --- 1. Load Processed Data --- 
    print("--- Loading processed data --- ")
    # (Assume files exist based on previous steps)
    data = np.load(processed_data_path)
    X_train_processed = data['X_train']
    y_train = data['y_train']
    X_test_processed = data['X_test']
    y_test = data['y_test']
    preprocessor = joblib.load(preprocessor_path)
    with open(feature_names_path, 'r') as f:
        processed_feature_names = [line.strip() for line in f.readlines()]
    print(f"Loaded data, preprocessor, and {len(processed_feature_names)} feature names.")
    if X_train_processed.shape[1] != len(processed_feature_names):
        print(f"Warning: Adjusting feature names list length.")
        processed_feature_names = processed_feature_names[:X_train_processed.shape[1]]

    # --- 2. Apply IQR Filtering (Optional) --- 
    if apply_iqr_filter:
        print("\n--- Applying IQR Filtering (on training data only) ---")
        lower_bound, upper_bound = calculate_iqr_bounds(X_train_processed)
        outlier_mask = ((X_train_processed < lower_bound) | (X_train_processed > upper_bound)).any(axis=1)
        X_train_filtered = X_train_processed[~outlier_mask]
        y_train_filtered = y_train[~outlier_mask]
        print(f"Filtered training data shape: X={X_train_filtered.shape}, y={y_train_filtered.shape}")
    else:
        print("\n--- Skipping IQR Filtering --- ")
        X_train_filtered = X_train_processed
        y_train_filtered = y_train

    # --- 3. Apply SMOTE (Optional) ---
    if apply_smote:
        print("\n--- Applying SMOTE --- ")
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_filtered, y_train_filtered)
        print(f"Resampled training data shape: X={X_train_resampled.shape}, y={y_train_resampled.shape}")
    else:
        print("\n--- Skipping SMOTE --- ")
        X_train_resampled = X_train_filtered
        y_train_resampled = y_train_filtered

    # --- 4. Define Hyperparameter Search Space --- 
    print("\n--- Defining Hyperparameter Search Space for RandomForest ---")
    param_dist = {
        'n_estimators': randint(50, 500),
        'max_depth': [None] + list(randint(10, 50).rvs(5)), # None + 5 random depths
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 11),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    # --- 5. Randomized Search CV --- 
    print(f"\n--- Starting RandomizedSearchCV (n_iter={n_iter_search}, cv={cv_folds}) ---")
    rf = RandomForestClassifier(random_state=random_state)
    # Use F1 score for optimization, as it balances precision and recall
    f1_scorer = make_scorer(f1_score)
    
    random_search = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_dist, 
        n_iter=n_iter_search, 
        cv=cv_folds, 
        scoring=f1_scorer, 
        n_jobs=-1, # Use all available CPU cores
        random_state=random_state,
        verbose=1 # Show progress
    )
    
    random_search.fit(X_train_resampled, y_train_resampled)
    
    print(f"Best Parameters found: {random_search.best_params_}")
    print(f"Best F1 Score (cross-validation): {random_search.best_score_:.4f}")
    
    best_rf_model = random_search.best_estimator_

    # --- 6. MLflow Logging for Tuning & Final Model --- 
    print(f"\n--- Starting MLflow Run for Tuned RandomForest ---")
    mlflow.set_experiment(mlflow_experiment_name)
    
    with mlflow.start_run(run_name="RandomForest_Tuned") as run:
        mlflow_run_id = run.info.run_id
        print(f"MLflow Run ID: {mlflow_run_id}")
        
        # Log data handling params
        mlflow.log_param("apply_iqr_filter", apply_iqr_filter)
        mlflow.log_param("apply_smote", apply_smote)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_param("tuning_n_iter", n_iter_search)
        mlflow.log_param("tuning_cv_folds", cv_folds)
        mlflow.log_param("tuning_scoring", "f1")

        # Log best hyperparameters found
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("best_cv_f1_score", random_search.best_score_)

        # --- 7. Evaluate Final Tuned Model --- 
        print("\n--- Evaluating Final Tuned Model on Test Set ---")
        final_metrics, final_clr, final_cm = evaluate_model(best_rf_model, X_test_processed, y_test)
        
        print(f"Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Precision: {final_metrics['precision']:.4f}")
        print(f"Recall: {final_metrics['recall']:.4f}")
        print(f"F1 Score: {final_metrics['f1_score']:.4f}")
        print(f"ROC AUC: {final_metrics['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(final_clr)
        print("Confusion Matrix:")
        print(final_cm)
        
        # Log final test metrics
        mlflow.log_metrics({f"test_{k}": v for k, v in final_metrics.items()})

        # Log reports/matrices as artifacts
        with open("tuned_rf_classification_report.txt", "w") as f:
            f.write(final_clr)
        mlflow.log_artifact("tuned_rf_classification_report.txt")
        os.remove("tuned_rf_classification_report.txt")
        
        with open("tuned_rf_confusion_matrix.txt", "w") as f:
             f.write(np.array2string(final_cm))
        mlflow.log_artifact("tuned_rf_confusion_matrix.txt")
        os.remove("tuned_rf_confusion_matrix.txt")

        # Log the final tuned model
        mlflow.sklearn.log_model(best_rf_model, "tuned_random_forest_model")
        print("\nLogged final tuned model to MLflow.")
        
        # Log the preprocessor and feature names again for this run
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
        mlflow.log_artifact(feature_names_path, artifact_path="features")
        print(f"Logged preprocessor and feature names to MLflow run.")
        
        # --- 8. Save Final Model Explicitly (Recommended) --- 
        final_model_save_path = os.path.join(os.path.dirname(__file__), '../models', 'final_churn_model.joblib')
        joblib.dump(best_rf_model, final_model_save_path)
        print(f"\nSaved final tuned model explicitly to {final_model_save_path}")

    print(f"\n--- MLflow Run ({mlflow_run_id}) completed. --- ")
    print(f"--- Hyperparameter Tuning Script finished successfully! --- ")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    processed_data_file = os.path.join(base_dir, '..', 'data', 'processed_data.npz')
    preprocessor_file = os.path.join(base_dir, '..', 'models', 'preprocessor.joblib')
    feature_names_file = os.path.join(base_dir, '..', 'data', 'processed_feature_names.txt')

    if not all(os.path.exists(f) for f in [processed_data_file, preprocessor_file, feature_names_file]):
        print("Error: One or more required files not found.")
        print("Please run the preprocessing script (02_preprocessing.py) first.")
    else:
        tune_log_final_model(
            processed_data_path=processed_data_file, 
            preprocessor_path=preprocessor_file,
            feature_names_path=feature_names_file,
            apply_iqr_filter=True, # Consistent with previous runs
            apply_smote=True,      # Consistent with previous runs
            n_iter_search=20,      # Number of iterations for RandomizedSearch
            cv_folds=5             # Number of CV folds
        ) 