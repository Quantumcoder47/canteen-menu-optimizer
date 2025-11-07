#!/usr/bin/env python3
"""
Canteen Menu Optimizer - Backend API

FastAPI backend for serving ML model predictions for dietary preferences.
Provides endpoints for health checks and predictions.

Author: AI Assistant
Date: 2024
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Canteen Menu Optimizer API",
    description="ML API for predicting student dietary preferences",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
feature_names = None

class StudentInput(BaseModel):
    """Input schema for student dietary preference prediction"""
    age: Optional[int] = Field(default=20, ge=16, le=30, description="Student age")
    height_cm: Optional[float] = Field(default=170.0, ge=140.0, le=200.0, description="Height in cm")
    weight_kg: Optional[float] = Field(default=65.0, ge=40.0, le=120.0, description="Weight in kg")
    spice_tolerance: Optional[int] = Field(default=5, ge=1, le=10, description="Spice tolerance (1-10)")
    sweet_tooth_level: Optional[int] = Field(default=5, ge=1, le=10, description="Sweet tooth level (1-10)")
    eating_out_per_week: Optional[int] = Field(default=3, ge=0, le=21, description="Eating out frequency per week")
    food_budget_per_meal: Optional[float] = Field(default=150.0, ge=50.0, le=1000.0, description="Food budget per meal in INR")
    cuisine_top1: Optional[str] = Field(default="Indian", description="Preferred cuisine")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 21,
                "height_cm": 175.0,
                "weight_kg": 70.0,
                "spice_tolerance": 7,
                "sweet_tooth_level": 6,
                "eating_out_per_week": 4,
                "food_budget_per_meal": 200.0,
                "cuisine_top1": "Indian"
            }
        }

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    predicted_preference: str
    confidence: str
    probability: float
    all_probabilities: Dict[str, float]
    business_insights: Dict[str, Any]

def load_model():
    """Load the trained model"""
    global model, feature_names
    
    try:
        # Try multiple paths for model file
        possible_paths = [
            Path("../model/canteen_prediction_model.joblib"),
            Path("model/canteen_prediction_model.joblib"),
            Path("canteen-App/model/canteen_prediction_model.joblib"),
            Path("../canteen-App/model/canteen_prediction_model.joblib"),
            Path("/opt/render/project/src/canteen-App/model/canteen_prediction_model.joblib"),
            Path("./canteen_prediction_model.joblib")
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            # List current directory for debugging
            logger.error(f"Current directory: {os.getcwd()}")
            logger.error(f"Directory contents: {os.listdir('.')}")
            raise FileNotFoundError("Model file not found in any expected location")
        
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        # Get feature names from the model pipeline
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            # This will be set when we make the first prediction
            feature_names = None
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_input(input_data: StudentInput) -> pd.DataFrame:
    """Preprocess input data to match model expectations"""
    
    # Convert input to dictionary
    data = input_data.dict()
    
    # Calculate BMI
    height_m = data['height_cm'] / 100
    bmi = data['weight_kg'] / (height_m ** 2)
    data['bmi'] = round(bmi, 2)
    
    # Create BMI category
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
    data['bmi_category'] = bmi_category
    
    # Create budget category
    budget = data['food_budget_per_meal']
    if budget <= 100:
        budget_category = "Low"
    elif budget <= 200:
        budget_category = "Medium"
    elif budget <= 500:
        budget_category = "High"
    else:
        budget_category = "Premium"
    data['budget_category'] = budget_category
    
    # Create eating frequency category
    eating_freq = data['eating_out_per_week']
    if eating_freq <= 2:
        freq_category = "Rare"
    elif eating_freq <= 5:
        freq_category = "Occasional"
    elif eating_freq <= 7:
        freq_category = "Frequent"
    else:
        freq_category = "Daily"
    data['eating_frequency_category'] = freq_category
    
    # Create engineered features
    data['spice_sweet_interaction'] = data['spice_tolerance'] * data['sweet_tooth_level']
    data['preference_intensity'] = (data['spice_tolerance'] + data['sweet_tooth_level']) / 2
    data['num_cuisines_recorded'] = 1  # Assuming one cuisine provided
    data['cuisine_diversity'] = 1
    
    # Create DataFrame with expected features
    expected_features = [
        'age', 'bmi', 'spice_tolerance', 'sweet_tooth_level',
        'eating_out_per_week', 'food_budget_per_meal',
        'num_cuisines_recorded', 'cuisine_diversity',
        'spice_sweet_interaction', 'preference_intensity',
        'cuisine_top1', 'bmi_category', 'budget_category', 'eating_frequency_category'
    ]
    
    # Create DataFrame with only expected features
    processed_data = {feature: data.get(feature, 0) for feature in expected_features}
    df = pd.DataFrame([processed_data])
    
    return df

def get_business_insights(prediction: str, probability: float, input_data: StudentInput) -> Dict[str, Any]:
    """Generate business insights based on prediction"""
    
    business_rules = {
        'Non-Veg': {
            'popular_items': ['Chicken Curry', 'Mutton Biryani', 'Fish Fry', 'Egg Curry'],
            'avg_cost': 150,
            'profit_margin': 0.35,
            'recommendations': ['Focus on protein-rich options', 'Offer spicy variants']
        },
        'Veg': {
            'popular_items': ['Dal Tadka', 'Paneer Curry', 'Veg Biryani', 'Chole Bhature'],
            'avg_cost': 100,
            'profit_margin': 0.40,
            'recommendations': ['Emphasize fresh vegetables', 'Offer healthy options']
        },
        'Vegan': {
            'popular_items': ['Quinoa Salad', 'Vegan Curry', 'Fruit Bowl', 'Smoothie'],
            'avg_cost': 120,
            'profit_margin': 0.45,
            'recommendations': ['Focus on plant-based proteins', 'Highlight nutritional benefits']
        },
        'Jain': {
            'popular_items': ['Jain Dal', 'Paneer without Onion', 'Jain Sabzi', 'Fruit Salad'],
            'avg_cost': 110,
            'profit_margin': 0.38,
            'recommendations': ['Ensure no root vegetables', 'Offer traditional Jain dishes']
        },
        'Eggitarian': {
            'popular_items': ['Egg Curry', 'Omelet', 'Egg Fried Rice', 'Scrambled Eggs'],
            'avg_cost': 80,
            'profit_margin': 0.50,
            'recommendations': ['Quick preparation items', 'Breakfast options']
        }
    }
    
    rules = business_rules.get(prediction, business_rules['Non-Veg'])
    
    # Determine confidence level
    if probability > 0.8:
        confidence_level = "Very High"
        reliability = "Highly reliable prediction"
    elif probability > 0.6:
        confidence_level = "High"
        reliability = "Reliable prediction"
    elif probability > 0.4:
        confidence_level = "Medium"
        reliability = "Moderately reliable prediction"
    else:
        confidence_level = "Low"
        reliability = "Low confidence - consider multiple options"
    
    # Budget compatibility
    budget_match = "High" if input_data.food_budget_per_meal >= rules['avg_cost'] else "Medium" if input_data.food_budget_per_meal >= rules['avg_cost'] * 0.8 else "Low"
    
    return {
        'popular_items': rules['popular_items'],
        'estimated_cost': rules['avg_cost'],
        'profit_margin': f"{rules['profit_margin']*100:.0f}%",
        'recommendations': rules['recommendations'],
        'confidence_level': confidence_level,
        'reliability': reliability,
        'budget_compatibility': budget_match,
        'spice_preference': "High" if input_data.spice_tolerance >= 7 else "Medium" if input_data.spice_tolerance >= 4 else "Low",
        'sweet_preference': "High" if input_data.sweet_tooth_level >= 7 else "Medium" if input_data.sweet_tooth_level >= 4 else "Low"
    }

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Canteen Menu Optimizer API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_dietary_preference(input_data: StudentInput):
    """Predict dietary preference for a student"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Preprocess input data
        processed_df = preprocess_input(input_data)
        
        logger.info(f"Making prediction for input: {input_data.dict()}")
        
        # Make prediction
        prediction = model.predict(processed_df)[0]
        probabilities = model.predict_proba(processed_df)[0]
        
        # Get class names
        class_names = model.named_steps['classifier'].classes_
        
        # Calculate confidence
        max_prob = np.max(probabilities)
        confidence = "High" if max_prob > 0.7 else "Medium" if max_prob > 0.5 else "Low"
        
        # Create probability dictionary
        prob_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
        
        # Generate business insights
        business_insights = get_business_insights(prediction, max_prob, input_data)
        
        logger.info(f"Prediction successful: {prediction} (confidence: {max_prob:.3f})")
        
        return PredictionResponse(
            predicted_preference=prediction,
            confidence=confidence,
            probability=float(max_prob),
            all_probabilities=prob_dict,
            business_insights=business_insights
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get model information
        info = {
            "model_type": type(model.named_steps['classifier']).__name__,
            "classes": model.named_steps['classifier'].classes_.tolist(),
            "n_features": model.named_steps['classifier'].n_features_in_,
            "feature_names": [
                'age', 'bmi', 'spice_tolerance', 'sweet_tooth_level',
                'eating_out_per_week', 'food_budget_per_meal',
                'num_cuisines_recorded', 'cuisine_diversity',
                'spice_sweet_interaction', 'preference_intensity',
                'cuisine_top1', 'bmi_category', 'budget_category', 'eating_frequency_category'
            ]
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)