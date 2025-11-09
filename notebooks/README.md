# Notebooks Directory

This directory contains the data analysis, model training, and business optimization notebooks for the Canteen Menu Optimizer project.

## 📁 Contents

### Data Files
- **`data.csv`** - Student dietary preference dataset with 111 records
  - Contains student demographics (age, height, weight, BMI)
  - Food preferences (spice tolerance, sweet tooth level)
  - Eating habits (frequency, budget, cuisine preferences)
  - Dietary preference labels (Non-Veg, Veg, Vegan, Jain, Eggitarian)

### Python Scripts
- **`canteen_business_optimizer.py`** - Complete business optimization system
  - Data preprocessing and feature engineering
  - ML model training (Random Forest Classifier)
  - Inventory recommendations generation
  - Customer segmentation analysis
  - Menu optimization reports
  - Business insights export

### Results Directory
- **`results/`** - Generated outputs and visualizations
  - Model performance metrics
  - Business insights reports
  - Inventory recommendations

## 🚀 Quick Start

### Run the Business Optimizer

```bash
# From project root
python notebooks/canteen_business_optimizer.py
```

This will:
1. Load and preprocess the dataset
2. Train the ML model (82.6% accuracy)
3. Generate predictions for all students
4. Create inventory recommendations
5. Analyze customer segments
6. Generate menu optimization reports
7. Save outputs to `business_insights/` directory

### Expected Outputs

After running, you'll find these files in `business_insights/`:

- **`canteen_prediction_model.joblib`** - Trained ML model
- **`inventory_recommendations.csv`** - Daily inventory planning
- **`customer_segments.json`** - Customer segmentation data
- **`menu_optimization_report.json`** - Menu optimization insights

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 82.6%
- **F1-Score (Macro)**: 0.19
- **Features**: 14 engineered features
- **Classes**: 5 dietary preferences

### Feature Engineering

The system creates these engineered features:
- BMI and BMI categories
- Budget categories (Low, Medium, High, Premium)
- Eating frequency categories (Rare, Occasional, Frequent, Daily)
- Spice-sweet interaction score
- Preference intensity score
- Cuisine diversity metrics

## 💼 Business Insights

### Inventory Recommendations
- Predicted demand per dietary preference
- Recommended quantities with safety stock
- Cost analysis and profit projections
- Preparation time estimates
- Seasonal adjustments

### Customer Segmentation
- Budget-based preferences
- Health-conscious preferences (BMI-based)
- Eating frequency patterns
- Targeted marketing insights

### Menu Optimization
- High-demand category identification
- Profit margin optimization
- Operational efficiency recommendations
- Quick preparation item suggestions

## 📈 Sample Results

Based on 111 students:
- **Daily Investment**: ₹19,130
- **Expected Revenue**: ₹29,911
- **Expected Profit**: ₹10,781
- **ROI**: 56.4%
- **Monthly Profit Potential**: ₹323,424

### Dietary Distribution
- Non-Veg: 87.4%
- Veg: 5.4%
- Jain: 3.6%
- Vegan: 1.8%
- Eggitarian: 1.8%

## 🔧 Requirements

```python
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install with:
```bash
pip install -r requirements.txt
```

## 📝 Data Schema

### Input Features
- `age` - Student age (16-30 years)
- `height_cm` - Height in centimeters
- `weight_kg` - Weight in kilograms
- `spice_tolerance` - Spice preference (1-10 scale)
- `sweet_tooth_level` - Sweet preference (1-10 scale)
- `eating_out_per_week` - Eating out frequency
- `food_budget_per_meal` - Budget per meal (INR)
- `cuisine_top1` - Primary cuisine preference

### Target Variable
- `dietary_pref` - Dietary preference category
  - Non-Veg
  - Veg
  - Vegan
  - Jain
  - Eggitarian

## 🎯 Use Cases

1. **Daily Menu Planning** - Predict demand and plan inventory
2. **Cost Optimization** - Minimize waste and maximize profit
3. **Customer Insights** - Understand preference patterns
4. **Seasonal Adjustments** - Adapt menu based on season
5. **Marketing Strategy** - Target specific customer segments

## 🔄 Workflow

```
Data Loading → Preprocessing → Feature Engineering
     ↓
Model Training → Validation → Prediction
     ↓
Business Analysis → Recommendations → Reports
     ↓
Export Insights → Integration with Frontend
```

## 📚 Additional Resources

- **Main Project**: See root `README.md` for full project documentation
- **Frontend**: See `canteen-App/frontend/` for Streamlit app
- **Deployment**: See `BACKEND_DEPLOYMENT.md` for deployment guide

## 🤝 Contributing

To add new features or improve the model:

1. Update `canteen_business_optimizer.py`
2. Test with the dataset
3. Validate model performance
4. Update this README with changes
5. Export new model to `canteen-App/model/`

## 📧 Contact

For questions or issues related to the notebooks:
- Check the main project README
- Review the code comments in `canteen_business_optimizer.py`
- Examine the generated business insights

---

**Last Updated**: November 2025  
**Model Version**: 1.0  
**Dataset Size**: 111 students
