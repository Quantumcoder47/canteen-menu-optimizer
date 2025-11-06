# 🍽️ Canteen Menu Optimizer - Complete Setup Guide

## 📋 Overview

This is a complete ML web application for predicting student dietary preferences to optimize canteen menu planning. The system consists of:

- **Backend API** (FastAPI) - Serves the ML model
- **Frontend Web App** (Streamlit) - User interface
- **Trained ML Model** - Random Forest classifier

## 🚀 Quick Start (Recommended)

### Step 1: Start the Backend API

```bash
# Option A: Use the batch script (Windows)
double-click start_backend.bat

# Option B: Manual start
cd canteen-App/backend
pip install -r requirements.txt
python main.py
```

**Backend will be available at:** http://localhost:8000

### Step 2: Start the Frontend

```bash
# Option A: Use the batch script (Windows)
double-click start_frontend.bat

# Option B: Manual start
cd canteen-App/frontend
pip install -r requirements.txt
streamlit run app.py
```

**Frontend will be available at:** http://localhost:8501

## 📁 Project Structure

```
canteen-App/
├── backend/                    # FastAPI backend
│   ├── main.py                # API endpoints and ML model serving
│   ├── requirements.txt       # Backend dependencies
│   └── __pycache__/          # Python cache (auto-generated)
├── frontend/                  # Streamlit frontend
│   ├── app.py                # User interface
│   ├── requirements.txt      # Frontend dependencies
│   └── .streamlit/           # Streamlit config (auto-generated)
├── model/                    # ML model files
│   └── canteen_prediction_model.joblib
├── start_backend.bat         # Windows script to start backend
├── start_frontend.bat        # Windows script to start frontend
├── README.md                 # Project documentation
└── SETUP_GUIDE.md           # This setup guide
```

## 🔧 Manual Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd canteen-App/backend
   ```

2. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn joblib pandas numpy scikit-learn pydantic python-multipart
   ```

3. **Start the backend server:**
   ```bash
   python main.py
   ```

4. **Verify backend is running:**
   - Open http://localhost:8000 in your browser
   - You should see: `{"message": "Canteen Menu Optimizer API", "status": "active"}`
   - API documentation: http://localhost:8000/docs

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd canteen-App/frontend
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit requests plotly pandas numpy
   ```

3. **Start the frontend:**
   ```bash
   streamlit run app.py
   ```

4. **Access the web application:**
   - Streamlit will automatically open http://localhost:8501
   - If not, manually navigate to that URL

## 🎯 Using the Application

### 1. Input Student Information

Fill out the form in the sidebar with:
- **Personal Details:** Age, height, weight
- **Food Preferences:** Spice tolerance, sweet tooth level
- **Eating Habits:** Frequency, budget, preferred cuisine

### 2. Get Predictions

Click "🔮 Predict Dietary Preference" to get:
- **Main prediction** with confidence level
- **Probability distribution** across all categories
- **Business insights** including popular items and recommendations

### 3. Export Results

Download prediction results as JSON for record-keeping or further analysis.

## 📊 API Endpoints

### Backend API (http://localhost:8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check |
| `/predict` | POST | Make dietary preference prediction |
| `/model-info` | GET | Get model information |
| `/docs` | GET | Interactive API documentation |

### Example API Usage

```python
import requests

# Make a prediction
data = {
    "age": 21,
    "height_cm": 175.0,
    "weight_kg": 70.0,
    "spice_tolerance": 7,
    "sweet_tooth_level": 6,
    "eating_out_per_week": 4,
    "food_budget_per_meal": 200.0,
    "cuisine_top1": "Indian"
}

response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()
print(f"Predicted preference: {result['predicted_preference']}")
```

## 🔍 Troubleshooting

### Common Issues

1. **"Backend API is not running" error:**
   - Ensure backend is started first
   - Check if port 8000 is available
   - Verify no firewall blocking

2. **"Model not loaded" error:**
   - Check if `canteen_prediction_model.joblib` exists in `model/` directory
   - Ensure all backend dependencies are installed

3. **Import errors:**
   - Install missing packages: `pip install <package_name>`
   - Use virtual environment to avoid conflicts

4. **Port already in use:**
   - Backend: Change port in `main.py` (line with `uvicorn.run`)
   - Frontend: Use `streamlit run app.py --server.port 8502`

### Verification Steps

1. **Check backend health:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Test prediction endpoint:**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"age": 21, "height_cm": 175, "weight_kg": 70, "spice_tolerance": 7, "sweet_tooth_level": 6, "eating_out_per_week": 4, "food_budget_per_meal": 200, "cuisine_top1": "Indian"}'
   ```

3. **Check frontend connection:**
   - Look for green "✅ Backend API is running successfully!" message
   - If red error, backend is not accessible

## 🎨 Features

### Backend Features
- **RESTful API** with automatic documentation
- **ML Model Serving** using joblib
- **Input Validation** with Pydantic models
- **Business Logic** for menu recommendations
- **Health Checks** and comprehensive error handling
- **CORS Support** for frontend integration

### Frontend Features
- **Interactive UI** with sliders and forms
- **Real-time Predictions** via API calls
- **Data Visualization** with Plotly charts
- **Business Insights** display
- **Export Functionality** for results
- **Responsive Design** with custom CSS

### ML Model Features
- **Random Forest Classifier** with preprocessing pipeline
- **Feature Engineering** (BMI calculation, categorization)
- **Class Imbalance Handling** with balanced weights
- **Confidence Scoring** for predictions
- **Business Rules Integration** for actionable insights

## 📈 Business Impact

### For Canteen Managers
- **Inventory Optimization:** Predict demand accurately (82.6% accuracy)
- **Cost Reduction:** Minimize food waste through better planning
- **Revenue Increase:** Serve items students actually want
- **Customer Satisfaction:** Data-driven menu decisions

### Key Metrics
- **Model Accuracy:** 82.6%
- **Daily ROI:** 56.4% (estimated)
- **Monthly Profit Potential:** ₹323,424 (estimated)
- **Prediction Categories:** Non-Veg, Veg, Vegan, Jain, Eggitarian

## 🔄 Development

### Adding New Features

1. **Backend changes:** Modify `backend/main.py`
2. **Frontend changes:** Modify `frontend/app.py`
3. **Model updates:** Replace `model/canteen_prediction_model.joblib`

### Testing

1. **Backend testing:**
   ```bash
   # Test all endpoints
   curl http://localhost:8000/
   curl http://localhost:8000/health
   curl http://localhost:8000/model-info
   ```

2. **Frontend testing:**
   - Fill form with various inputs
   - Check prediction results
   - Verify visualizations

## 📞 Support

### Getting Help

1. **Check logs:** Both backend and frontend show detailed error messages
2. **API documentation:** Visit http://localhost:8000/docs for interactive API testing
3. **Verify setup:** Follow troubleshooting steps above

### System Requirements

- **OS:** Windows, macOS, or Linux
- **Python:** 3.8 or higher
- **RAM:** 2GB minimum (4GB recommended)
- **Storage:** 500MB for dependencies and model

---

**🎉 You're all set! Enjoy using the Canteen Menu Optimizer!**