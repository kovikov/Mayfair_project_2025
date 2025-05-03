# Mayfair_project_2025

Employee Churn Prediction Project.

This project aims to predict employee churn using machine learning.
It includes data preprocessing, model training, evaluation, deployment via FastAPI, and automated retraining.

## Project Structure

```
├── data/
│   └── data.csv
├── models/
├── notebooks/
├── scripts/
├── src/
│   └── main.py
├── tests/
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/kovikov/Mayfair_project_2025.git
   cd Mayfair_project_2025
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   # source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

(Instructions on how to run the analysis, train the model, and use the API will be added here.)

## Deployment

The model is deployed as a REST API using FastAPI and Docker, hosted on Render.com.

## Retraining

The model is automatically retrained quarterly using [Apache Airflow / GitHub Actions - specify later]. 