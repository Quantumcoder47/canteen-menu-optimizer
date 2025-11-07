#!/bin/bash
# Startup script for Render deployment

# Copy model file to backend directory if it exists
if [ -f "../model/canteen_prediction_model.joblib" ]; then
    echo "Copying model file to backend directory..."
    cp ../model/canteen_prediction_model.joblib ./canteen_prediction_model.joblib
fi

# Start the FastAPI server with Gunicorn
echo "Starting FastAPI server with Gunicorn..."
gunicorn -c gunicorn_conf.py main:app
