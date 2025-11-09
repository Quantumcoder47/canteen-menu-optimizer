#!/usr/bin/env python3
"""
Canteen Business Optimizer - Production Ready System

This system provides:
- Real-time dietary preference prediction
- Inventory optimization recommendations
- Menu planning insights
- Cost-benefit analysis
- Seasonal trend analysis
- Customer segmentation

Author: Karan Prabhat
Date: 2025
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter, defaultdict
import json
from datetime import datetime, timedelta
import joblib

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Configuration
RANDOM_STATE = 42
warnings.filterwarnings("ignore")

class CanteenBusinessOptimizer:
    """Complete business optimization system for canteen management"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.class_names = []
        self.business_rules = self.load_business_rules()
        
    def load_business_rules(self):
        """Load business rules and constraints"""
        return {
            'dietary_preferences': {
                'Non-Veg': {
                    'popular_items': ['Chicken Curry', 'Mutton Biryani', 'Fish Fry', 'Egg Curry'],
                    'avg_cost_per_meal': 150,
                    'profit_margin': 0.35,
                    'preparation_time': 45,
                    'shelf_life_hours': 4
                },
                'Veg': {
                    'popular_items': ['Dal Tadka', 'Paneer Curry', 'Veg Biryani', 'Chole Bhature'],
                    'avg_cost_per_meal': 100,
                    'profit_margin': 0.40,
                    'preparation_time': 30,
                    'shelf_life_hours': 6
                },
                'Vegan': {
                    'popular_items': ['Quinoa Salad', 'Vegan Curry', 'Fruit Bowl', 'Smoothie'],
                    'avg_cost_per_meal': 120,
                    'profit_margin': 0.45,
                    'preparation_time': 25,
                    'shelf_life_hours': 3
                },
                'Jain': {
                    'popular_items': ['Jain Dal', 'Paneer without Onion', 'Jain Sabzi', 'Fruit Salad'],
                    'avg_cost_per_meal': 110,
                    'profit_margin': 0.38,
                    'preparation_time': 35,
                    'shelf_life_hours': 5
                },
                'Eggitarian': {
                    'popular_items': ['Egg Curry', 'Omelet', 'Egg Fried Rice', 'Scrambled Eggs'],
                    'avg_cost_per_meal': 80,
                    'profit_margin': 0.50,
                    'preparation_time': 15,
                    'shelf_life_hours': 3
                }
            },
            'peak_hours': {
                'breakfast': {'start': 8, 'end': 10, 'multiplier': 1.2},
                'lunch': {'start': 12, 'end': 14, 'multiplier': 2.0},
                'snacks': {'start': 16, 'end': 18, 'multiplier': 1.5},
                'dinner': {'start': 19, 'end': 21, 'multiplier': 1.8}
            },
            'seasonal_factors': {
                'summer': {'veg_boost': 1.3, 'vegan_boost': 1.5, 'non_veg_reduction': 0.8},
                'winter': {'non_veg_boost': 1.4, 'veg_stable': 1.0, 'vegan_reduction': 0.7},
                'monsoon': {'comfort_food_boost': 1.6, 'fresh_food_reduction': 0.6}
            }
        }
    
    def load_and_preprocess_data(self, filepath="notebooks/data.csv"):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing data...")
        
        df = pd.read_csv(filepath)
        
        # Clean column names
        def clean_col(c):
            return (str(c).strip().replace("\n", " ").replace("  ", " ")
                   .lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_"))
        
        df.columns = [clean_col(c) for c in df.columns]
        
        # Preprocessing steps
        if "age" in df.columns:
            df["age"] = df["age"].astype(str).str.extract(r"(\d+)")[0]
            df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int64")
        
        # Handle height and weight
        height_col = "height_cm" if "height_cm" in df.columns else "height"
        weight_col = "weight_kg" if "weight_kg" in df.columns else "weight"
        
        for col in (height_col, weight_col):
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r"[^\d\.]", "", regex=True), 
                    errors="coerce"
                )
        
        # Calculate BMI and categories
        if height_col in df.columns and weight_col in df.columns:
            height_in_m = df[height_col] / 100 if "cm" in height_col else df[height_col]
            df["bmi"] = df[weight_col] / (height_in_m ** 2)
            df["bmi"] = df["bmi"].round(2)
            df["bmi_category"] = pd.cut(
                df["bmi"], 
                bins=[0, 18.5, 25, 30, float('inf')], 
                labels=["Underweight", "Normal", "Overweight", "Obese"]
            )
        
        # Handle cuisine preferences
        cuisine_cols = [c for c in ("cuisine_top1", "cuisine_top2", "cuisine_top3") if c in df.columns]
        for c in cuisine_cols:
            df[c] = df[c].astype(str).replace("nan", pd.NA).replace("", pd.NA).str.strip()
        
        if cuisine_cols:
            df["num_cuisines_recorded"] = df[cuisine_cols].notna().sum(axis=1)
            df["cuisine_diversity"] = df[cuisine_cols].apply(
                lambda row: len(set(row.dropna())), axis=1
            )
        
        # Numeric features
        numeric_cols = ["spice_tolerance", "sweet_tooth_level", "eating_out_per_week", "food_budget_per_meal"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace(r"[^\d\.]", "", regex=True), 
                    errors="coerce"
                )
        
        # Feature engineering
        if "spice_tolerance" in df.columns and "sweet_tooth_level" in df.columns:
            df["spice_sweet_interaction"] = df["spice_tolerance"] * df["sweet_tooth_level"]
            df["preference_intensity"] = (df["spice_tolerance"] + df["sweet_tooth_level"]) / 2
        
        if "food_budget_per_meal" in df.columns:
            df["budget_category"] = pd.cut(
                df["food_budget_per_meal"],
                bins=[0, 100, 200, 500, float('inf')],
                labels=["Low", "Medium", "High", "Premium"]
            )
        
        if "eating_out_per_week" in df.columns:
            df["eating_frequency_category"] = pd.cut(
                df["eating_out_per_week"],
                bins=[0, 2, 5, 7, float('inf')],
                labels=["Rare", "Occasional", "Frequent", "Daily"]
            )
        
        # Target variable
        target_col = "dietary_pref" if "dietary_pref" in df.columns else "dietary_preference"
        if target_col in df.columns:
            df[target_col] = df[target_col].astype(str).str.strip().str.title()
            df = df[df[target_col].notna() & (df[target_col] != "Nan")]
        
        df = df.drop_duplicates().reset_index(drop=True)
        
        return df, target_col
    
    def train_model(self, df, target_col):
        """Train the prediction model"""
        print("Training prediction model...")
        
        # Define features
        numerical_features = [
            c for c in [
                "age", "bmi", "spice_tolerance", "sweet_tooth_level",
                "eating_out_per_week", "food_budget_per_meal",
                "num_cuisines_recorded", "cuisine_diversity",
                "spice_sweet_interaction", "preference_intensity"
            ] if c in df.columns
        ]
        
        categorical_features = [
            c for c in [
                "cuisine_top1", "bmi_category", "budget_category", "eating_frequency_category"
            ] if c in df.columns
        ]
        
        self.feature_names = numerical_features + categorical_features
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features)
            ],
            remainder="drop"
        )
        
        # Create and train model
        self.model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200, 
                class_weight="balanced", 
                random_state=RANDOM_STATE
            ))
        ])
        
        # Prepare data
        X = df[self.feature_names]
        y = df[target_col]
        
        # Train model
        self.model.fit(X, y)
        self.class_names = self.model.named_steps['classifier'].classes_
        
        # Evaluate model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"Model Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Macro: {f1_macro:.4f}")
        
        return accuracy, f1_macro
    
    def predict_dietary_preference(self, student_data):
        """Predict dietary preference for a student"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Convert to DataFrame
        if isinstance(student_data, dict):
            student_df = pd.DataFrame([student_data])
        else:
            student_df = student_data.copy()
        
        # Make prediction
        prediction = self.model.predict(student_df[self.feature_names])
        probabilities = self.model.predict_proba(student_df[self.feature_names])
        
        # Get confidence
        max_prob = np.max(probabilities, axis=1)[0]
        confidence = "High" if max_prob > 0.7 else "Medium" if max_prob > 0.5 else "Low"
        
        return {
            'predicted_preference': prediction[0],
            'confidence': confidence,
            'probability': max_prob,
            'all_probabilities': dict(zip(self.class_names, probabilities[0]))
        }
    
    def generate_inventory_recommendations(self, predictions_df, date=None, meal_type="lunch"):
        """Generate inventory recommendations based on predictions"""
        print(f"\n=== INVENTORY RECOMMENDATIONS FOR {meal_type.upper()} ===\n")
        
        if date is None:
            date = datetime.now()
        
        # Get seasonal factor
        month = date.month
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "summer"
        elif month in [6, 7, 8, 9]:
            season = "monsoon"
        else:
            season = "summer"
        
        seasonal_factors = self.business_rules['seasonal_factors'][season]
        
        # Count predicted preferences
        preference_counts = predictions_df['predicted_preference'].value_counts()
        total_students = len(predictions_df)
        
        recommendations = []
        
        for preference, count in preference_counts.items():
            if preference in self.business_rules['dietary_preferences']:
                rules = self.business_rules['dietary_preferences'][preference]
                
                # Apply seasonal adjustments
                if preference == 'Veg' and 'veg_boost' in seasonal_factors:
                    count = int(count * seasonal_factors['veg_boost'])
                elif preference == 'Non-Veg' and 'non_veg_boost' in seasonal_factors:
                    count = int(count * seasonal_factors['non_veg_boost'])
                elif preference == 'Vegan' and 'vegan_boost' in seasonal_factors:
                    count = int(count * seasonal_factors['vegan_boost'])
                
                # Calculate quantities
                base_quantity = count
                safety_stock = max(2, int(count * 0.15))  # 15% safety stock
                total_quantity = base_quantity + safety_stock
                
                # Calculate costs
                cost_per_meal = rules['avg_cost_per_meal']
                total_cost = total_quantity * cost_per_meal
                expected_revenue = total_quantity * cost_per_meal / (1 - rules['profit_margin'])
                expected_profit = expected_revenue - total_cost
                
                recommendation = {
                    'dietary_preference': preference,
                    'predicted_demand': count,
                    'recommended_quantity': total_quantity,
                    'safety_stock': safety_stock,
                    'popular_items': rules['popular_items'],
                    'cost_per_meal': cost_per_meal,
                    'total_cost': total_cost,
                    'expected_revenue': expected_revenue,
                    'expected_profit': expected_profit,
                    'profit_margin': rules['profit_margin'],
                    'preparation_time_minutes': rules['preparation_time'],
                    'shelf_life_hours': rules['shelf_life_hours'],
                    'season': season,
                    'seasonal_adjustment': True if preference in ['Veg', 'Non-Veg', 'Vegan'] else False
                }
                
                recommendations.append(recommendation)
        
        # Sort by expected profit
        recommendations.sort(key=lambda x: x['expected_profit'], reverse=True)
        
        # Display recommendations
        total_cost = sum(r['total_cost'] for r in recommendations)
        total_revenue = sum(r['expected_revenue'] for r in recommendations)
        total_profit = sum(r['expected_profit'] for r in recommendations)
        
        print(f"Total Students: {total_students}")
        print(f"Season: {season.title()}")
        print(f"Date: {date.strftime('%Y-%m-%d')}")
        print(f"Meal Type: {meal_type.title()}")
        print()
        
        for rec in recommendations:
            print(f"{rec['dietary_preference']}:")
            print(f"  Predicted Demand: {rec['predicted_demand']} students")
            print(f"  Recommended Quantity: {rec['recommended_quantity']} meals")
            print(f"  Popular Items: {', '.join(rec['popular_items'][:3])}")
            print(f"  Total Cost: ₹{rec['total_cost']:,.0f}")
            print(f"  Expected Revenue: ₹{rec['expected_revenue']:,.0f}")
            print(f"  Expected Profit: ₹{rec['expected_profit']:,.0f}")
            print(f"  Preparation Time: {rec['preparation_time_minutes']} minutes")
            print()
        
        print(f"SUMMARY:")
        print(f"  Total Investment: ₹{total_cost:,.0f}")
        print(f"  Expected Revenue: ₹{total_revenue:,.0f}")
        print(f"  Expected Profit: ₹{total_profit:,.0f}")
        print(f"  Overall Margin: {(total_profit/total_revenue)*100:.1f}%")
        
        return recommendations
    
    def analyze_customer_segments(self, df):
        """Analyze customer segments for targeted marketing"""
        print("\n=== CUSTOMER SEGMENTATION ANALYSIS ===\n")
        
        segments = {}
        
        # Budget-based segmentation
        if 'food_budget_per_meal' in df.columns:
            budget_segments = df.groupby('budget_category')['dietary_pref'].value_counts()
            segments['budget'] = budget_segments
            
            print("Budget-Based Preferences:")
            for budget_cat in df['budget_category'].unique():
                if pd.notna(budget_cat):
                    subset = df[df['budget_category'] == budget_cat]
                    top_pref = subset['dietary_pref'].mode().iloc[0] if len(subset) > 0 else "Unknown"
                    avg_budget = subset['food_budget_per_meal'].mean()
                    print(f"  {budget_cat} Budget (₹{avg_budget:.0f}): Prefers {top_pref}")
            print()
        
        # BMI-based segmentation
        if 'bmi_category' in df.columns:
            bmi_segments = df.groupby('bmi_category')['dietary_pref'].value_counts()
            segments['bmi'] = bmi_segments
            
            print("Health-Conscious Preferences:")
            for bmi_cat in df['bmi_category'].unique():
                if pd.notna(bmi_cat):
                    subset = df[df['bmi_category'] == bmi_cat]
                    top_pref = subset['dietary_pref'].mode().iloc[0] if len(subset) > 0 else "Unknown"
                    print(f"  {bmi_cat} BMI: Prefers {top_pref}")
            print()
        
        # Eating frequency segmentation
        if 'eating_frequency_category' in df.columns:
            freq_segments = df.groupby('eating_frequency_category')['dietary_pref'].value_counts()
            segments['frequency'] = freq_segments
            
            print("Eating Frequency Preferences:")
            for freq_cat in df['eating_frequency_category'].unique():
                if pd.notna(freq_cat):
                    subset = df[df['eating_frequency_category'] == freq_cat]
                    top_pref = subset['dietary_pref'].mode().iloc[0] if len(subset) > 0 else "Unknown"
                    print(f"  {freq_cat} Eaters: Prefers {top_pref}")
            print()
        
        return segments
    
    def generate_menu_optimization_report(self, df, predictions_df):
        """Generate comprehensive menu optimization report"""
        print("\n=== MENU OPTIMIZATION REPORT ===\n")
        
        # Current vs Predicted distribution
        actual_dist = df['dietary_pref'].value_counts(normalize=True) * 100
        predicted_dist = predictions_df['predicted_preference'].value_counts(normalize=True) * 100
        
        print("Dietary Preference Distribution:")
        print("Preference\tActual\tPredicted\tDifference")
        print("-" * 50)
        
        for pref in actual_dist.index:
            actual_pct = actual_dist.get(pref, 0)
            predicted_pct = predicted_dist.get(pref, 0)
            diff = predicted_pct - actual_pct
            print(f"{pref}\t\t{actual_pct:.1f}%\t{predicted_pct:.1f}%\t\t{diff:+.1f}%")
        
        print()
        
        # Menu recommendations
        print("MENU OPTIMIZATION RECOMMENDATIONS:")
        print()
        
        # High-demand items
        top_preferences = predicted_dist.head(3)
        print("1. FOCUS ON HIGH-DEMAND CATEGORIES:")
        for pref, pct in top_preferences.items():
            if pref in self.business_rules['dietary_preferences']:
                items = self.business_rules['dietary_preferences'][pref]['popular_items']
                print(f"   • {pref} ({pct:.1f}%): {', '.join(items[:2])}")
        print()
        
        # Profit optimization
        print("2. PROFIT OPTIMIZATION:")
        profit_margins = {}
        for pref in predicted_dist.index:
            if pref in self.business_rules['dietary_preferences']:
                margin = self.business_rules['dietary_preferences'][pref]['profit_margin']
                demand = predicted_dist[pref]
                profit_score = margin * demand
                profit_margins[pref] = profit_score
        
        sorted_profit = sorted(profit_margins.items(), key=lambda x: x[1], reverse=True)
        for pref, score in sorted_profit[:3]:
            margin = self.business_rules['dietary_preferences'][pref]['profit_margin']
            print(f"   • {pref}: {margin*100:.0f}% margin (Priority Score: {score:.1f})")
        print()
        
        # Operational efficiency
        print("3. OPERATIONAL EFFICIENCY:")
        prep_times = {}
        for pref in predicted_dist.index:
            if pref in self.business_rules['dietary_preferences']:
                prep_time = self.business_rules['dietary_preferences'][pref]['preparation_time']
                demand = predicted_dist[pref]
                prep_times[pref] = prep_time
        
        sorted_prep = sorted(prep_times.items(), key=lambda x: x[1])
        print("   Quick Preparation Items (< 30 min):")
        for pref, time in sorted_prep:
            if time < 30:
                print(f"     • {pref}: {time} minutes")
        print()
        
        return {
            'actual_distribution': actual_dist.to_dict(),
            'predicted_distribution': predicted_dist.to_dict(),
            'profit_rankings': dict(sorted_profit),
            'preparation_times': prep_times
        }
    
    def save_business_insights(self, recommendations, segments, optimization_report):
        """Save business insights to files"""
        print("\n=== SAVING BUSINESS INSIGHTS ===\n")
        
        os.makedirs("business_insights", exist_ok=True)
        
        # Save inventory recommendations
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df.to_csv("business_insights/inventory_recommendations.csv", index=False)
        
        # Save customer segments
        with open("business_insights/customer_segments.json", "w") as f:
            # Convert pandas objects to serializable format
            serializable_segments = {}
            for key, value in segments.items():
                if hasattr(value, 'to_dict'):
                    # Convert tuple keys to strings for JSON serialization
                    dict_value = value.to_dict()
                    serializable_dict = {}
                    for k, v in dict_value.items():
                        if isinstance(k, tuple):
                            serializable_dict[str(k)] = v
                        else:
                            serializable_dict[str(k)] = v
                    serializable_segments[key] = serializable_dict
                else:
                    serializable_segments[key] = str(value)
            json.dump(serializable_segments, f, indent=2)
        
        # Save optimization report
        with open("business_insights/menu_optimization_report.json", "w") as f:
            json.dump(optimization_report, f, indent=2)
        
        # Save model
        if self.model:
            joblib.dump(self.model, "business_insights/canteen_prediction_model.joblib")
        
        print("✓ inventory_recommendations.csv - Daily inventory planning")
        print("✓ customer_segments.json - Customer segmentation analysis")
        print("✓ menu_optimization_report.json - Menu optimization insights")
        print("✓ canteen_prediction_model.joblib - Trained prediction model")
        
        return True

def main():
    """Main execution function"""
    print("=== CANTEEN BUSINESS OPTIMIZER ===\n")
    
    # Initialize optimizer
    optimizer = CanteenBusinessOptimizer()
    
    # Load and preprocess data
    df, target_col = optimizer.load_and_preprocess_data()
    
    # Train model
    accuracy, f1_score = optimizer.train_model(df, target_col)
    
    # Generate predictions for all students
    X = df[optimizer.feature_names]
    predictions = []
    
    for idx, row in X.iterrows():
        pred_result = optimizer.predict_dietary_preference(row.to_dict())
        predictions.append({
            'student_id': idx,
            'predicted_preference': pred_result['predicted_preference'],
            'confidence': pred_result['confidence'],
            'probability': pred_result['probability']
        })
    
    predictions_df = pd.DataFrame(predictions)
    
    # Generate business insights
    recommendations = optimizer.generate_inventory_recommendations(predictions_df)
    segments = optimizer.analyze_customer_segments(df)
    optimization_report = optimizer.generate_menu_optimization_report(df, predictions_df)
    
    # Save insights
    optimizer.save_business_insights(recommendations, segments, optimization_report)
    
    # Final summary
    total_profit = sum(r['expected_profit'] for r in recommendations)
    total_investment = sum(r['total_cost'] for r in recommendations)
    roi = (total_profit / total_investment) * 100 if total_investment > 0 else 0
    
    print(f"\n=== BUSINESS IMPACT SUMMARY ===\n")
    print(f"Model Accuracy: {accuracy:.1%}")
    print(f"Total Students: {len(df)}")
    print(f"Daily Investment Required: ₹{total_investment:,.0f}")
    print(f"Expected Daily Profit: ₹{total_profit:,.0f}")
    print(f"Return on Investment: {roi:.1f}%")
    print(f"Monthly Profit Potential: ₹{total_profit * 30:,.0f}")
    
    return optimizer

if __name__ == "__main__":
    optimizer = main()