import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

print("Simple House Price Prediction")
print("=" * 60)

class SimpleHousePredictor:
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.features = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
            'floors', 'waterfront', 'view', 'condition', 'grade',
            'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
            'sqft_living15', 'sqft_lot15'
        ]
        
        print(f"Predictor initialized with file: {csv_file}")
    
    def load_data(self):
        print("\n STEP 1: LOADING DATA")
        print("-" * 30)
        
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f" Data loaded successfully!")
            print(f"    Total houses: {len(self.data):,}")
            print(f"    Total features: {len(self.data.columns)}")
            
            
            print(f"\n First 3 houses in the dataset:")
            print(self.data.head(3))
            
        except Exception as e:
            print(f" Error loading data: {e}")
            return False
        
        return True
    
    def explore_data(self):
        
        print("\n STEP 2: EXPLORING DATA")
        print("-" * 30)
        
        if self.data is None:
            print(" No data loaded. Call load_data() first.")
            return
        
        
        print(f" Dataset Overview:")
        print(f"   Shape: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns")
        
        
        missing_values = self.data.isnull().sum()
        total_missing = missing_values.sum()
        
        if total_missing > 0:
            print(f"  Missing values found: {total_missing:,}")
            print("   Missing values by column:")
            for col, missing in missing_values.items():
                if missing > 0:
                    percentage = (missing / len(self.data)) * 100
                    print(f"     {col}: {missing:,} ({percentage:.1f}%)")
        else:
            print(" No missing values found!")
        
        
        if 'price' in self.data.columns:
            price_stats = self.data['price'].describe()
            print(f"\n Price Statistics:")
            print(f"   Average price: ${price_stats['mean']:,.2f}")
            print(f"   Median price: ${price_stats['50%']:,.2f}")
            print(f"   Cheapest house: ${price_stats['min']:,.2f}")
            print(f"   Most expensive: ${price_stats['max']:,.2f}")
            print(f"   Price range: ${price_stats['max'] - price_stats['min']:,.2f}")
        
        
        print(f"\n House Features:")
        for feature in ['bedrooms', 'bathrooms', 'sqft_living', 'grade']:
            if feature in self.data.columns:
                stats = self.data[feature].describe()
                print(f"   {feature}: avg={stats['mean']:.1f}, min={stats['min']:.0f}, max={stats['max']:.0f}")
    
    def clean_data(self):
        """Step 3: Clean and prepare the data"""
        print("\n STEP 3: CLEANING DATA")
        print("-" * 30)
        
        if self.data is None:
            print(" No data loaded. Call load_data() first.")
            return
        
        original_size = len(self.data)
        print(f" Starting with {original_size:,} houses")
        
        
        if 'price' in self.data.columns:
            self.data = self.data.dropna(subset=['price'])
            print(f" Removed houses with missing prices")
        
        
        for feature in self.features:
            if feature in self.data.columns:
                self.data[feature] = self.data[feature].fillna(0)
        
        
        if 'price' in self.data.columns:
            
            price_99th = self.data['price'].quantile(0.99)
            before_outliers = len(self.data)
            self.data = self.data[self.data['price'] < price_99th]
            after_outliers = len(self.data)
            removed = before_outliers - after_outliers
            
            print(f" Removed {removed:,} extremely expensive houses (>{price_99th:,.0f})")
        
        
        for feature in self.features:
            if feature in self.data.columns:
                self.data[feature] = pd.to_numeric(self.data[feature], errors='coerce').fillna(0)
        
        final_size = len(self.data)
        print(f" Final dataset: {final_size:,} houses")
        print(f" Removed: {original_size - final_size:,} houses ({(original_size - final_size)/original_size*100:.1f}%)")
        
        
        if 'price' in self.data.columns:
            final_price_stats = self.data['price'].describe()
            print(f" Final price range: ${final_price_stats['min']:,.0f} - ${final_price_stats['max']:,.0f}")
    
    def prepare_features(self):
        
        print("\n STEP 4: PREPARING FEATURES")
        print("-" * 30)
        
        if self.data is None:
            print(" No data loaded. Call load_data() and clean_data() first.")
            return
        
        
        available_features = [f for f in self.features if f in self.data.columns]
        print(f" Using {len(available_features)} features: {available_features}")
        
        
        X = self.data[available_features]
        y = self.data['price']
        
        print(f" Features shape: {X.shape}")
        print(f" Target shape: {y.shape}")
        
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f" Data split successfully:")
        print(f"    Training set: {len(self.X_train):,} houses")
        print(f"    Test set: {len(self.X_test):,} houses")
        print(f"    Features: {self.X_train.shape[1]}")
    
    def train_model(self):
        """Step 5: Train the machine learning model"""
        print("\n STEP 5: TRAINING MODEL")
        print("-" * 30)
        
        if self.X_train is None:
            print(" Features not prepared. Call prepare_features() first.")
            return
        
        
        print(" Creating Linear Regression model...")
        self.model = LinearRegression()
        
        print(" Training the model...")
        self.model.fit(self.X_train, self.y_train)
        
        print(" Model trained successfully!")
        
        
        print("\n TESTING THE MODEL")
        print("-" * 30)
        
        
        train_predictions = self.model.predict(self.X_train)
        test_predictions = self.model.predict(self.X_test)
        
        
        train_r2 = r2_score(self.y_train, train_predictions)
        test_r2 = r2_score(self.y_test, test_predictions)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_predictions))
        
        train_mae = mean_absolute_error(self.y_train, train_predictions)
        test_mae = mean_absolute_error(self.y_test, test_predictions)
        
        
        print(f" Model Performance:")
        print(f"    Training R² Score: {train_r2:.3f} ({train_r2*100:.1f}%)")
        print(f"    Test R² Score: {test_r2:.3f} ({test_r2*100:.1f}%)")
        print(f"    Training RMSE: ${train_rmse:,.0f}")
        print(f"    Test RMSE: ${test_rmse:,.0f}")
        print(f"    Training MAE: ${train_mae:,.0f}")
        print(f"    Test MAE: ${test_mae:,.0f}")
        
       
        r2_difference = train_r2 - test_r2
        if r2_difference > 0.1:
            print(f"  Warning: Model might be overfitting (R² difference: {r2_difference:.3f})")
        else:
            print(f" Model looks good! (R² difference: {r2_difference:.3f})")
        
        
        print(f"\n FEATURE IMPORTANCE")
        print("-" * 30)
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'coefficient': self.model.coef_,
            'importance': np.abs(self.model.coef_)
        }).sort_values('importance', ascending=False)
        
        print("Top 5 most important features:")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
            direction = "increases" if row['coefficient'] > 0 else "decreases"
            print(f"   {i}. {row['feature']:<15} | {direction} price by ${row['coefficient']:,.0f}")
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
    
    def save_model(self, filename='house_price_model.pkl'):
        
        print(f"\n STEP 6: SAVING MODEL")
        print("-" * 30)
        
        if self.model is None:
            print(" No model to save. Train the model first.")
            return
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.model, f)
            print(f" Model saved as '{filename}'")
        except Exception as e:
            print(f" Error saving model: {e}")
    
    def load_model(self, filename='house_price_model.pkl'):
        
        print(f"\n LOADING MODEL")
        print("-" * 30)
        
        try:
            with open(filename, 'rb') as f:
                self.model = pickle.load(f)
            print(f" Model loaded from '{filename}'")
        except Exception as e:
            print(f" Error loading model: {e}")
    
    def predict_price(self, house_features):
        
        if self.model is None:
            print(" No model loaded. Train or load a model first.")
            return None
        
        try:
            
            if isinstance(house_features, dict):
                
                for feature in self.features:
                    if feature not in house_features:
                        house_features[feature] = 0
                
                
                feature_df = pd.DataFrame([house_features])[self.features]
            else:
                feature_df = house_features
            
            
            prediction = self.model.predict(feature_df)
            return float(prediction[0])
            
        except Exception as e:
            print(f" Error making prediction: {e}")
            return None
    
    def create_simple_visualization(self):
        
        print("\n CREATING VISUALIZATIONS")
        print("-" * 30)
        
        if self.data is None:
            print(" No data loaded.")
            return
        
        try:
            
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('House Price Analysis - Student Version', fontsize=16, fontweight='bold')
            
           
            axes[0, 0].hist(self.data['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('House Price Distribution')
            axes[0, 0].set_xlabel('Price ($)')
            axes[0, 0].set_ylabel('Number of Houses')
            axes[0, 0].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            
            
            axes[0, 1].scatter(self.data['sqft_living'], self.data['price'], alpha=0.5, color='green')
            axes[0, 1].set_title('Price vs Living Area')
            axes[0, 1].set_xlabel('Living Area (sq ft)')
            axes[0, 1].set_ylabel('Price ($)')
            axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            
            bedroom_prices = self.data.groupby('bedrooms')['price'].mean()
            axes[1, 0].bar(bedroom_prices.index, bedroom_prices.values, color='orange', alpha=0.7)
            axes[1, 0].set_title('Average Price by Bedrooms')
            axes[1, 0].set_xlabel('Number of Bedrooms')
            axes[1, 0].set_ylabel('Average Price ($)')
            axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            
            grade_prices = self.data.groupby('grade')['price'].mean()
            axes[1, 1].plot(grade_prices.index, grade_prices.values, marker='o', color='purple', linewidth=2)
            axes[1, 1].set_title('Average Price by Grade')
            axes[1, 1].set_xlabel('House Grade')
            axes[1, 1].set_ylabel('Average Price ($)')
            axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig('house_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(" Visualization saved as 'house_analysis.png'")
            
        except Exception as e:
            print(f" Error creating visualization: {e}")


def main():
    
    print(" STARTING HOUSE PRICE PREDICTION")
    print("=" * 60)
    print("This will teach you how machine learning works step by step!")
    print("=" * 60)
    
    
    predictor = SimpleHousePredictor('house_data.csv')
    
    
    if not predictor.load_data():
        print(" Failed to load data. Please check if 'house_data.csv' exists.")
        return
    
    
    predictor.explore_data()
    
    
    predictor.clean_data()
    
    
    predictor.prepare_features()
    
    
    metrics = predictor.train_model()
    
    
    predictor.save_model()
    
    
    predictor.create_simple_visualization()
    
    
    print("\n EXAMPLE PREDICTION")
    print("-" * 30)
    example_house = {
        'bedrooms': 3,
        'bathrooms': 2,
        'sqft_living': 2000,
        'sqft_lot': 5000,
        'floors': 1,
        'waterfront': 0,
        'view': 0,
        'condition': 3,
        'grade': 7,
        'sqft_above': 2000,
        'sqft_basement': 0,
        'yr_built': 2000,
        'yr_renovated': 0,
        'sqft_living15': 2000,
        'sqft_lot15': 5000
    }
    
    predicted_price = predictor.predict_price(example_house)
    if predicted_price:
        print(f" Example house: 3 bed, 2 bath, 2000 sqft")
        print(f" Predicted price: ${predicted_price:,.2f}")
    
   
    if metrics:
        print(f"\n Final Model Performance:")
        print(f"   Accuracy: {metrics['test_r2']*100:.1f}%")
        print(f"   Average Error: ${metrics['test_mae']:,.0f}")
    
    
    
    return predictor


if __name__ == "__main__":
     
    predictor = main()