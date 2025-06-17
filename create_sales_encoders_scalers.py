import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

class SalesFeatureTransformer:
    """
    A class to transform sales prediction features using fitted encoders and scaler
    """
    
    def __init__(self, product_type_encoder, marketing_channel_encoder, season_encoder, scaler):
        self.product_type_encoder = product_type_encoder
        self.marketing_channel_encoder = marketing_channel_encoder
        self.season_encoder = season_encoder
        self.scaler = scaler
    
    def transform(self, data):
        """
        Transform input data using the fitted encoders and scaler
        """
        # Encode categorical features
        product_type_encoded = self.product_type_encoder.transform([data['product_type']])[0]
        marketing_channel_encoded = self.marketing_channel_encoder.transform([data['marketing_channel']])[0]
        season_encoded = self.season_encoder.transform([data['season']])[0]
        
        # Scale numerical features
        numerical_data = np.array([[data['ad_budget'], data['unit_price'], data['units_sold']]])
        numerical_scaled = self.scaler.transform(numerical_data)[0]
        
        # Combine all features
        features = [
            product_type_encoded,
            marketing_channel_encoded, 
            season_encoded,
            numerical_scaled[0],  # ad_budget
            numerical_scaled[1],  # unit_price
            numerical_scaled[2]   # units_sold
        ]
        
        return features

def create_sales_encoders_scalers():
    """
    Create encoders and scalers for sales prediction data based on the training dataset
    """
    
    # Load the training data
    data_path = "models/sales/large_sales_revenue_prediction_data.csv"
    df = pd.read_csv(data_path)
    
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nSample data:")
    print(df.head())
    
    # Create output directory if it doesn't exist
    output_dir = "models/sales"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create Label Encoders for categorical variables
    print("\n=== Creating Label Encoders ===")
    
    # Product Type Encoder
    product_type_encoder = LabelEncoder()
    product_type_encoder.fit(df['product_type'])
    print("Product types:", product_type_encoder.classes_)
    joblib.dump(product_type_encoder, os.path.join(output_dir, 'product_type_encoder.pkl'))
    print("✓ Saved product_type_encoder.pkl")
    
    # Marketing Channel Encoder
    marketing_channel_encoder = LabelEncoder()
    marketing_channel_encoder.fit(df['marketing_channel'])
    print("Marketing channels:", marketing_channel_encoder.classes_)
    joblib.dump(marketing_channel_encoder, os.path.join(output_dir, 'marketing_channel_encoder.pkl'))
    print("✓ Saved marketing_channel_encoder.pkl")
    
    # Season Encoder
    season_encoder = LabelEncoder()
    season_encoder.fit(df['season'])
    print("Seasons:", season_encoder.classes_)
    joblib.dump(season_encoder, os.path.join(output_dir, 'season_encoder.pkl'))
    print("✓ Saved season_encoder.pkl")
    
    # 2. Create Standard Scaler for numerical variables
    print("\n=== Creating Standard Scaler ===")
    
    # Prepare numerical features for scaling
    numerical_features = ['ad_budget', 'unit_price', 'units_sold']
    X_numerical = df[numerical_features]
    
    # Create and fit the scaler
    scaler = StandardScaler()
    scaler.fit(X_numerical)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(output_dir, 'sales_scaler.pkl'))
    print("✓ Saved sales_scaler.pkl")
    
    # Print scaling statistics
    print("\nScaling statistics:")
    for i, feature in enumerate(numerical_features):
        print(f"{feature}:")
        print(f"  Mean: {scaler.mean_[i]:.2f}")
        print(f"  Scale: {scaler.scale_[i]:.2f}")
    
    # 3. Create a combined feature transformer
    print("\n=== Creating Combined Feature Transformer ===")
    
    # Create the feature transformer
    feature_transformer = SalesFeatureTransformer(
        product_type_encoder, 
        marketing_channel_encoder, 
        season_encoder, 
        scaler
    )
    
    # Save the feature transformer
    joblib.dump(feature_transformer, os.path.join(output_dir, 'feature_transformer.pkl'))
    print("✓ Saved feature_transformer.pkl")
    
    # 4. Test the encoders and scalers
    print("\n=== Testing Encoders and Scalers ===")
    
    # Test with sample data
    test_data = {
        'product_type': 'Laptop',
        'marketing_channel': 'Social Media',
        'season': 'Summer',
        'ad_budget': 10000,
        'unit_price': 1000,
        'units_sold': 20
    }
    
    print("Test input:", test_data)
    
    # Test individual encoders
    print(f"Product type encoded: {product_type_encoder.transform([test_data['product_type']])[0]}")
    print(f"Marketing channel encoded: {marketing_channel_encoder.transform([test_data['marketing_channel']])[0]}")
    print(f"Season encoded: {season_encoder.transform([test_data['season']])[0]}")
    
    # Test scaler
    test_numerical = np.array([[test_data['ad_budget'], test_data['unit_price'], test_data['units_sold']]])
    scaled_numerical = scaler.transform(test_numerical)
    print(f"Scaled numerical features: {scaled_numerical[0]}")
    
    # Test combined transformation
    transformed_features = feature_transformer.transform(test_data)
    print(f"Combined transformed features: {transformed_features}")
    
    # 5. Create a summary file
    summary = {
        'categorical_features': {
            'product_type': {
                'encoder_file': 'product_type_encoder.pkl',
                'classes': product_type_encoder.classes_.tolist()
            },
            'marketing_channel': {
                'encoder_file': 'marketing_channel_encoder.pkl',
                'classes': marketing_channel_encoder.classes_.tolist()
            },
            'season': {
                'encoder_file': 'season_encoder.pkl',
                'classes': season_encoder.classes_.tolist()
            }
        },
        'numerical_features': {
            'scaler_file': 'sales_scaler.pkl',
            'features': numerical_features,
            'means': scaler.mean_.tolist(),
            'scales': scaler.scale_.tolist()
        },
        'feature_transformer_file': 'feature_transformer.pkl',
        'data_info': {
            'total_samples': len(df),
            'features': df.columns.tolist(),
            'target': 'sales_revenue'
        }
    }
    
    # Save summary as JSON
    import json
    with open(os.path.join(output_dir, 'encoders_scalers_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("✓ Saved encoders_scalers_summary.json")
    
    print("\n=== Summary ===")
    print("Created files:")
    print("- product_type_encoder.pkl")
    print("- marketing_channel_encoder.pkl") 
    print("- season_encoder.pkl")
    print("- sales_scaler.pkl")
    print("- feature_transformer.pkl")
    print("- encoders_scalers_summary.json")
    print(f"\nAll files saved to: {output_dir}")

if __name__ == "__main__":
    create_sales_encoders_scalers() 