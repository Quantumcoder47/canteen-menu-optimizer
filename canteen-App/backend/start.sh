#!/bin/bash
# Startup script for Render deployment

# Copy model file to backend directory if it exists
if [ -f "../model/canteen_prediction_model.joblib" ]; then
    echo "Copying model file to backend directory..."
    cp ../model/canteen_prediction_model.joblib ./canteen_prediction_model.joblib
fi

# Start the FastAPI server
echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
