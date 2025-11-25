from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)


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
            print("Model not found. Creating a simple model...")
            
            from sklearn.linear_model import LinearRegression
            
            
            data = pd.read_csv('house_data.csv')
            
            
            feature_columns = [
                'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                'floors', 'waterfront', 'view', 'condition', 'grade',
                'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                'sqft_living15', 'sqft_lot15'
            ]
            
            
            data = data.dropna(subset=feature_columns + ['price'])
            
            
            price_99th = data['price'].quantile(0.99)
            data = data[data['price'] < price_99th]
            
            X = data[feature_columns]
            y = data['price']
            
            
            predictor = LinearRegression()
            predictor.fit(X, y)
            
            
            with open('house_price_model.pkl', 'wb') as f:
                pickle.dump(predictor, f)
            print("Simple model created and saved!")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        predictor = None

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        if predictor is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        
        data = request.get_json()
        
        
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
        
        
        feature_df = pd.DataFrame([features])
        
        
        prediction = predictor.predict(feature_df)
        
        return jsonify({
            'predicted_price': round(prediction[0], 2),
            'formatted_price': f"${prediction[0]:,.2f}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analysis')
def analysis():
    
    return render_template('analysis.html')

@app.route('/about')
def about():
    
    return render_template('about.html')

if __name__ == '__main__':
    print("Starting House Price Prediction Web App...")
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
