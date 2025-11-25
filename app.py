from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from house_price_prediction import HousePricePredictor
except ImportError:
    print("Warning: Could not import HousePricePredictor. Some features may not work.")

app = Flask(__name__)

# Global variable to store the predictor
predictor = None

def load_model():
    """Load the trained model"""
    global predictor
    try:
        if os.path.exists('house_price_model.pkl'):
            with open('house_price_model.pkl', 'rb') as f:
                predictor = pickle.load(f)
            print("Model loaded successfully!")
        else:
            print("Model not found. Training new model...")
            predictor_obj = HousePricePredictor('house_data.csv')
            predictor_obj.preprocess_data()
            predictor_obj.prepare_features()
            predictor_obj.train_model()
            predictor_obj.save_model()
            predictor = predictor_obj.model
    except Exception as e:
        print(f"Error loading model: {e}")
        predictor = None

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict house price"""
    try:
        if predictor is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Get form data
        data = request.get_json()
        
        # Prepare features
        features = {
            'bedrooms': float(data.get('bedrooms', 0)),
            'bathrooms': float(data.get('bathrooms', 0)),
            'sqft_living': float(data.get('sqft_living', 0)),
            'sqft_lot': float(data.get('sqft_lot', 0)),
            'floors': float(data.get('floors', 0)),
            'waterfront': int(data.get('waterfront', 0)),
            'view': int(data.get('view', 0)),
            'condition': int(data.get('condition', 0)),
            'grade': int(data.get('grade', 0)),
            'sqft_above': float(data.get('sqft_above', 0)),
            'sqft_basement': float(data.get('sqft_basement', 0)),
            'yr_built': int(data.get('yr_built', 0)),
            'yr_renovated': int(data.get('yr_renovated', 0)),
            'sqft_living15': float(data.get('sqft_living15', 0)),
            'sqft_lot15': float(data.get('sqft_lot15', 0))
        }
        
        # Create DataFrame
        feature_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = predictor.predict(feature_df)
        
        return jsonify({
            'predicted_price': round(prediction, 2),
            'formatted_price': f"${prediction:,.2f}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analysis')
def analysis():
    """Analysis page"""
    return render_template('analysis.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    print("🏠 Starting House Price Prediction Web App...")
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
