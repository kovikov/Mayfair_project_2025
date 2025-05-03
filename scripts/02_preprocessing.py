import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # For saving the preprocessor

def preprocess_data(data_path, output_dir='../data', test_size=0.2, random_state=42):
    """Loads, preprocesses the employee data, and saves the results."""
    
    # --- 1. Load Data --- 
    try:
        df = pd.read_csv(data_path)
        print(f"--- Data loaded successfully from {data_path} ---")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please ensure the file exists.")
        return None

    # --- 2. Initial Cleaning & Separation --- 
    print("\n--- Performing initial cleaning --- ")
    # Drop Employee ID
    if 'EMPLOYEE_ID' in df.columns:
        df_processed = df.drop('EMPLOYEE_ID', axis=1)
        print("Dropped EMPLOYEE_ID column.")
    else:
        df_processed = df.copy()
        print("EMPLOYEE_ID column not found.")
        
    # Correct potential leading/trailing spaces in column names
    df_processed.columns = df_processed.columns.str.strip()
    print("Stripped whitespace from column names.")
    
    # Separate features and target
    target_col = 'QUIT_THE_COMPANY'
    if target_col not in df_processed.columns:
        print(f"Error: Target column '{target_col}' not found.")
        return None
        
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]
    print(f"Separated features (X) and target (y: '{target_col}').")

    # --- 3. Define Feature Types --- 
    print("\n--- Identifying feature types --- ")
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features_nominal = ['DEPARTMENTS', 'JOB_ROLE']
    categorical_features_ordinal = ['SALARY']
    binary_features = ['WORK_ACCIDENT', 'PROMOTION_LAST_5YEARS', 'REMOTE_WORK']

    # Ensure identified categorical features actually exist in X
    categorical_features_nominal = [col for col in categorical_features_nominal if col in X.columns]
    categorical_features_ordinal = [col for col in categorical_features_ordinal if col in X.columns]
    
    # Remove binary features from numerical list as they don't need scaling (usually)
    # or require specific handling depending on the model.
    numerical_features = [col for col in numerical_features if col not in binary_features]
    
    print(f"Numerical Features (to scale): {numerical_features}")
    print(f"Binary Features (already 0/1): {binary_features}")
    print(f"Nominal Categorical Features (to one-hot encode): {categorical_features_nominal}")
    print(f"Ordinal Categorical Features (to ordinal encode): {categorical_features_ordinal}")

    # --- 4. Train-Test Split --- 
    print("\n--- Splitting data into training and testing sets --- ")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train set size: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set size: X={X_test.shape}, y={y_test.shape}")

    # --- 5. Create Preprocessing Pipelines --- 
    print("\n--- Defining preprocessing steps --- ")
    
    # Pipeline for numerical features: scale them
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # Pipeline for nominal categorical features: one-hot encode them
    nominal_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Pipeline for ordinal categorical features: ordinal encode them
    # Define the order for salary
    salary_order = ['low', 'medium', 'high']
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(categories=[salary_order]))])
    
    # Note: Binary features often don't need transformation, but could be included if needed.
    # We are leaving them out of the ColumnTransformer for now.
    
    # Combine preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('nom', nominal_transformer, categorical_features_nominal),
            ('ord', ordinal_transformer, categorical_features_ordinal),
            # ('bin', 'passthrough', binary_features) # Option to explicitly pass through binary
        ],
        remainder='passthrough' # Keep other columns (like binary) untouched
        )
        
    # --- 6. Apply Preprocessing --- 
    print("\n--- Applying preprocessing to train and test sets --- ")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print("Preprocessing applied.")
    
    # Get feature names after transformation (important for interpretability/MLflow logging)
    try:
        feature_names_out = preprocessor.get_feature_names_out()
        print(f"Shape of processed training data: {X_train_processed.shape}")
        # print("Feature names after processing:", feature_names_out)
    except Exception as e:
        print(f"Could not get feature names: {e}")
        feature_names_out = [] # Fallback

    # Convert processed arrays back to DataFrames (optional, but can be useful)
    # X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names_out, index=X_train.index)
    # X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_out, index=X_test.index)
    
    # --- 7. Save Processed Data & Preprocessor --- 
    print("\n--- Saving processed data and preprocessor --- ")
    processed_data_path = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(processed_data_path, exist_ok=True)
    
    # Save processed data (consider formats like parquet for efficiency)
    # Saving as numpy arrays along with target variables
    np.savez(os.path.join(processed_data_path, 'processed_data.npz'), 
             X_train=X_train_processed, y_train=y_train, 
             X_test=X_test_processed, y_test=y_test)
    print(f"Saved processed data to {os.path.join(processed_data_path, 'processed_data.npz')}")

    # Save the preprocessor pipeline
    preprocessor_path = os.path.join(os.path.dirname(__file__), '../models', 'preprocessor.joblib')
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessor pipeline to {preprocessor_path}")
    
    # Save feature names
    feature_names_path = os.path.join(processed_data_path, 'processed_feature_names.txt')
    with open(feature_names_path, 'w') as f:
        for name in feature_names_out:
            f.write(f"{name}\n")
    print(f"Saved processed feature names to {feature_names_path}")

    print("\n--- Preprocessing script finished successfully! ---")
    return preprocessor, X_train_processed, X_test_processed, y_train, y_test, feature_names_out

if __name__ == "__main__":
    # Define the data path relative to the script location
    base_dir = os.path.dirname(__file__)
    input_data_path = os.path.join(base_dir, '..', 'data', 'data.csv')
    output_data_dir = os.path.join(base_dir, '..', 'data') # Save processed data in data/ 
    
    preprocess_data(data_path=input_data_path, output_dir=output_data_dir) 