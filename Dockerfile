# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and necessary model artifacts
COPY ./src /app/src
COPY ./models/preprocessor.joblib /app/models/preprocessor.joblib
COPY ./models/final_churn_model.joblib /app/models/final_churn_model.joblib

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# Use Gunicorn for production deployments (recommended for Render)
# Ensure gunicorn is in requirements.txt if using this CMD
# CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.main:app", "--bind", "0.0.0.0:8000"]
# Development command (using uvicorn directly):
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] 