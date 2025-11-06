# Canteen Menu Optimizer - ML Web Application

A complete machine learning web application for predicting student dietary preferences to optimize canteen menu planning and inventory management.

## 🏗️ Architecture

```
canteen-App/
├── backend/                # FastAPI backend server
│   ├── main.py             # API endpoints and ML model serving
│   └── requirements.txt    # Backend dependencies
├── frontend/               # Streamlit frontend application
│   ├── app.py              # User interface and API client
│   └── requirements.txt    # Frontend dependencies
├── model/                  # Trained ML model
│   └── canteen_prediction_model.joblib
└── README.md               # This file
```

## 🚀 Quick Start

### Set-Up

1. **Backend Setup:**
   ```bash
   cd canteen-App/backend
   pip install -r requirements.txt
   python main.py
   ```
   Backend will be available at: http://localhost:8000

2. **Frontend Setup:**
   ```bash
   cd canteen-App/frontend
   pip install -r requirements.txt
   streamlit run app.py
   ```
   Frontend will be available at: http://localhost:8501

## 📋 Features

### Backend (FastAPI)
- **RESTful API** with automatic documentation
- **ML Model Serving** using joblib
- **Input Validation** with Pydantic models
- **Business Logic** for menu recommendations
- **Health Checks** and error handling
- **CORS Support** for frontend integration

### Frontend (Streamlit)
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

## 🎯 Usage

1. **Input Student Data:**
   - Personal details (age, height, weight)
   - Food preferences (spice tolerance, sweet tooth)
   - Eating habits (frequency, budget, cuisine)

2. **Get Predictions:**
   - Dietary preference prediction
   - Confidence scores
   - Probability distribution

3. **Business Insights:**
   - Popular menu items
   - Cost estimates
   - Profit margins
   - Customer recommendations

## 📊 API Endpoints

### Backend API (http://localhost:8000)

- `GET /` - Root endpoint with API info
- `GET /health` - Health check
- `POST /predict` - Make dietary preference prediction
- `GET /model-info` - Get model information
- `GET /docs` - Interactive API documentation

### Example API Request:
```json
{
  "age": 21,
  "height_cm": 175.0,
  "weight_kg": 70.0,
  "spice_tolerance": 7,
  "sweet_tooth_level": 6,
  "eating_out_per_week": 4,
  "food_budget_per_meal": 200.0,
  "cuisine_top1": "Indian"
}
```

### Example API Response:
```json
{
  "predicted_preference": "Non-Veg",
  "confidence": "High",
  "probability": 0.85,
  "all_probabilities": {
    "Non-Veg": 0.85,
    "Veg": 0.10,
    "Vegan": 0.03,
    "Jain": 0.01,
    "Eggitarian": 0.01
  },
  "business_insights": {
    "popular_items": ["Chicken Curry", "Mutton Biryani", "Fish Fry"],
    "estimated_cost": 150,
    "profit_margin": "35%",
    "recommendations": ["Focus on protein-rich options", "Offer spicy variants"]
  }
}
```

## 🛠️ Technical Details

### Dependencies

**Backend:**
- FastAPI - Modern web framework for APIs
- Uvicorn - ASGI server
- Pydantic - Data validation
- Joblib - Model serialization
- Pandas/NumPy - Data processing
- Scikit-learn - ML model

**Frontend:**
- Streamlit - Web app framework
- Requests - HTTP client
- Plotly - Interactive charts
- Pandas - Data manipulation

### Model Information
- **Type:** Random Forest Classifier
- **Features:** 14 engineered features
- **Classes:** Non-Veg, Veg, Vegan, Jain, Eggitarian
- **Accuracy:** ~82.6%
- **Preprocessing:** StandardScaler, OneHotEncoder

## 🔧 Troubleshooting

### Common Issues:

1. **Backend not starting:**
   - Check if port 8000 is available
   - Ensure all dependencies are installed
   - Verify model file exists

2. **Frontend connection error:**
   - Ensure backend is running first
   - Check backend URL in frontend code
   - Verify firewall settings

3. **Model loading error:**
   - Check model file path in backend
   - Ensure model file is not corrupted
   - Verify scikit-learn version compatibility

### Logs and Debugging:
- Backend logs appear in console
- Frontend errors shown in Streamlit interface
- Check browser console for additional errors

## 📈 Business Impact

### For Canteen Managers:
- **Inventory Optimization:** Predict demand accurately
- **Cost Reduction:** Minimize food waste
- **Revenue Increase:** Better menu planning
- **Customer Satisfaction:** Serve preferred items

### Key Metrics:
- **Daily Investment:** ₹19,130 (estimated)
- **Expected Profit:** ₹10,781 (56.4% ROI)
- **Monthly Potential:** ₹323,424
- **Accuracy:** 82.6% prediction accuracy

## 🔮 Future Enhancements

- **Real-time Analytics Dashboard**
- **Seasonal Trend Analysis**
- **Multi-location Support**
- **Mobile App Integration**
- **Advanced ML Models** (Deep Learning)
- **Feedback Loop** for model improvement

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