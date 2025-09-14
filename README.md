# Canteen Menu Optimizer

**Project:** ML Mini Project — Classification Challenge 
<br>
**Author:** Karan Prabhat

## Overview

This project predicts students' dietary preferences (Veg, Non-Veg, Vegan, etc.) using survey features such as cuisine choices, spice tolerance, sweet tooth level, BMI, and more. The workflow includes data cleaning, exploratory data analysis (EDA), feature engineering, model building, evaluation, and feature importance analysis.

## Requirements
See `requirements.txt`.

## Workflow

1. **Data Loading & Cleaning**
   - Load raw survey data from `canteen_data.csv`.
   - Clean column names and values for consistency.
   - Handle missing values and convert datatypes.
   - Calculate BMI: $\mathrm{BMI} = \frac{\text{weight (kg)}}{\text{height (m)}^2}$
   - Save cleaned data to `canteen_data_clean.csv`.

2. **Exploratory Data Analysis (EDA)**
   - Visualize dietary preference distribution.
   - Plot numeric feature distributions (BMI, spice tolerance, sweet tooth level, age).
   - Analyze cuisine popularity.
   - Generate correlation heatmaps for numeric features.
   - Visualize feature distributions by dietary preference (violin/box plots).

3. **Feature Engineering**
   - Create new features (e.g., number of cuisines recorded, spice-sweet interaction).
   - Encode categorical variables (One-Hot Encoding).
   - Scale numerical features.

4. **Model Building**
   - Split data into train/test sets (stratified).
   - Build ML pipelines using scikit-learn:
     - Logistic Regression (baseline)
     - Random Forest (with GridSearchCV for hyperparameter tuning)
   - Handle class imbalance using class weights.

5. **Model Evaluation**
   - Evaluate models using accuracy, macro F1-score, and classification reports.
   - Plot confusion matrix for test set predictions.

6. **Feature Importance**
   - Analyze Random Forest feature importances.
   - Compute permutation importances for robustness.
   - Visualize top features.

7. **Saving Results**
   - Save final trained model as `final_model_randomforest.joblib`.
   - Save cleaned dataset as `canteen_data_clean.csv`.
   - All plots are saved in the `plots/` directory.

## Usage

- Run all cells in `Canteen_Menu_Optimizer.ipynb` from top to bottom.
- Outputs (plots, model, cleaned data) are saved automatically.

## Files

- `canteen_data.csv` - Raw survey data
- `canteen_data_clean.csv` - Cleaned data after Preprocessing
- `Canteen_Menu_Optimizer.ipynb` - Main notebook with code and analysis
- `final_model_randomforest.joblib` - Saved trained model
- `plots/` - Directory containing all generated visualizations

## Insights

- The model predicts Non-Veg dietary preference accurately but struggles with minority classes due to dataset imbalance.
- Feature importance analysis helps identify key factors influencing dietary choices.
- Results can help canteen managers optimize menu planning and reduce food waste.

## Contact
prabhatkaran47@gmail.com
