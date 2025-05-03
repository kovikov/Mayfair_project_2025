import pandas as pd
import os

# Define the data path relative to the script location
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')

# Load the dataset
try:
    df = pd.read_csv(data_path)
    print(f"--- Data loaded successfully from {data_path} ---")
except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}. Please ensure the file exists.")
    df = None

if df is not None:
    print("\n--- First 5 rows of the dataset: ---")
    print(df.head())
    
    print("\n--- Dataset Info: ---")
    # Use buffer to capture info() output as it prints directly
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    print(s)
    
    print("\n--- Missing values per column: ---")
    print(df.isnull().sum())
    
    print("\n--- Descriptive Statistics (Numerical Features): ---")
    print(df.describe())
    
    print("\n--- Descriptive Statistics (Categorical Features): ---")
    try:
        # describe(include='object') might fail if there are no object columns
        print(df.describe(include='object'))
    except ValueError:
        print("No categorical features found.") 