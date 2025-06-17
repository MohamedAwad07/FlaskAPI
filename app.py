from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import json
from datetime import datetime
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for models and preprocessors
recommendation_model = None
popular_products = None
spam_detection_model = None
spam_scaler = None
spam_encoders = {}
sales_prediction_model = None
sales_preprocessors = {}

def load_models():
    """Load all ML models from their respective subdirectories"""
    global recommendation_model, popular_products, spam_detection_model, spam_scaler, spam_encoders, sales_prediction_model, sales_preprocessors
    try:
        # Recommendation models
        rec_dir = os.path.join("models", "recomndation")
        rec_model_path = os.path.join(rec_dir, "svd_collaborative_model.pkl")
        pop_products_path = os.path.join(rec_dir, "popular_products.pkl")
        
        try:
            if os.path.exists(rec_model_path):
                recommendation_model = joblib.load(rec_model_path)
                logging.info("Loaded SVD recommendation model.")
            else:
                logging.warning("SVD recommendation model not found.")
        except ImportError as e:
            if "surprise" in str(e):
                logging.warning("Surprise library not available. Recommendation model will use fallback logic.")
            else:
                logging.error(f"Error loading recommendation model: {e}")
        
        try:
            if os.path.exists(pop_products_path):
                popular_products = joblib.load(pop_products_path)
                logging.info("Loaded popular products list.")
            else:
                logging.warning("Popular products list not found.")
        except Exception as e:
            logging.error(f"Error loading popular products: {e}")



        # Spam detection model and preprocessors
        spam_dir = os.path.join("models", "spam-3", "spam")
        spam_model_path = os.path.join(spam_dir, "model.pkl")
        scaler_path = os.path.join(spam_dir, "scaler.pkl")  # Updated to correct filename
        
        if os.path.exists(spam_model_path):
            try:
                spam_detection_model = joblib.load(spam_model_path)
                logging.info("Loaded spam detection model (Random Forest).")
            except Exception as e:
                logging.error(f"Error loading spam model: {e}")
        else:
            logging.warning("Spam detection model not found.")
            
        if os.path.exists(scaler_path):
            try:
                spam_scaler = joblib.load(scaler_path)
                logging.info("Loaded spam scaler (StandardScaler).")
            except Exception as e:
                logging.error(f"Error loading spam scaler: {e}")
        else:
            logging.warning("Spam scaler (scaler.pkl) not found.")
            
        # Load encoders for spam detection
        encoders = {
            "Platform_Interaction_encoder.pkl": "Platform_Interaction_encoder",
            "Sales_Consistency.pkl": "Sales_Consistency_encoder", 
            "Label.pkl": "Label_encoder"
        }
        
        for enc_file, enc_name in encoders.items():
            enc_path = os.path.join(spam_dir, enc_file)
            if os.path.exists(enc_path):
                try:
                    spam_encoders[enc_name] = joblib.load(enc_path)
                    logging.info(f"Loaded spam encoder: {enc_name}")
                except Exception as e:
                    logging.error(f"Error loading encoder {enc_name}: {e}")
            else:
                logging.warning(f"Spam encoder not found: {enc_file}")



        # Sales prediction model and preprocessors
        sales_dir = os.path.join("models", "sales")
        sales_model_path = os.path.join(sales_dir, "sales.pkl")
        #xgboost_model_path = os.path.join(sales_dir, "XGBoost.pkl")
        
        # Try to load sales model (either sales.pkl or XGBoost.pkl)
        if os.path.exists(sales_model_path):
            try:
                sales_prediction_model = joblib.load(sales_model_path)
                logging.info("Loaded sales prediction model (sales.pkl).")
            except Exception as e:
                logging.error(f"Error loading sales model: {e}")
                sales_prediction_model = None
        else:
            logging.warning("Sales prediction model not found (sales.pkl).")
            
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")

# Load models when module is imported
load_models()

def validate_spam_input(data):
    """Validate input for spam detection"""
    required_fields = [
        "Profile_Completeness", "Sales_Consistency", "Customer_Feedback",
        "Transaction_History", "Platform_Interaction"
    ]
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        
        # Check if field is numeric (Profile_Completeness, Customer_Feedback, Transaction_History)
        if field in ["Profile_Completeness", "Customer_Feedback", "Transaction_History"]:
            if not isinstance(data[field], (int, float)):
                return False, f"Field {field} must be a number"
            if data[field] < 0 or data[field] > 1:
                return False, f"Field {field} must be between 0 and 1"
        
        # Check if field is categorical (Sales_Consistency, Platform_Interaction)
        elif field in ["Sales_Consistency", "Platform_Interaction"]:
            if not isinstance(data[field], str):
                return False, f"Field {field} must be a string"
            if not data[field].strip():
                return False, f"Field {field} cannot be empty"
    
    return True, "Valid input"

def validate_sales_input(data):
    """Validate input for sales prediction"""
    required_fields = ["product_type", "season", "marketing_channel", "ad_budget", "unit_price", "units_sold"]
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        
        # Validate string fields
        if field in ["product_type", "season", "marketing_channel"]:
            if not isinstance(data[field], str):
                return False, f"Field {field} must be a string"
            if not data[field].strip():
                return False, f"Field {field} cannot be empty"
        
        # Validate numeric fields
        elif field in ["ad_budget", "unit_price", "units_sold"]:
            try:
                if field in ["ad_budget", "unit_price"]:
                    float(data[field])
                else:  # units_sold
                    int(data[field])
            except (ValueError, TypeError):
                return False, f"Field {field} must be a valid number"
    
    return True, "Valid input"

@app.route('/', methods=['GET'])
def welcome():
    """Welcome page with API information"""
    return jsonify({
        "message": "Welcome to Flask AI/ML API Backend",
        "description": "A REST API serving three machine learning models for recommendation, spam detection, and sales prediction",
        "version": "1.0.0",
        "endpoints": {
            "health": {
                "url": "/health",
                "method": "GET",
                "description": "Check API status and model loading status"
            },
            "recommendation": {
                "url": "/recommend",
                "method": "POST",
                "description": "Get product recommendations for customers",
                "example_input": {"customer_id": "12345"}
            },
            "spam_detection": {
                "url": "/detect-spam",
                "method": "POST",
                "description": "Detect spam based on user profile metrics",
                "example_input": {
                    "Profile_Completeness": 0.8,
                    "Sales_Consistency": "high",
                    "Customer_Feedback": 0.9,
                    "Transaction_History": 0.7,
                    "Platform_Interaction": "medium"
                }
            },
            "sales_prediction": {
                "url": "/predict-sales",
                "method": "POST",
                "description": "Predict sales revenue based on product and marketing data",
                "example_input": {
                    "product_type": "electronics",
                    "season": "summer",
                    "marketing_channel": "social_media",
                    "ad_budget": 5000,
                    "unit_price": 99.99,
                    "units_sold": 100
                }
            }
        },
        "models_status": {
            "recommendation": recommendation_model is not None or popular_products is not None,
            "spam_detection": spam_detection_model is not None,
            "sales_prediction": sales_prediction_model is not None
        },
        "documentation": "See /health for detailed model status and API_Documentation.md for complete documentation",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "recommendation": recommendation_model is not None or popular_products is not None,
            "spam_detection": spam_detection_model is not None,
            "sales_prediction": sales_prediction_model is not None
        }
    })


@app.route('/recommend', methods=['POST'])
def recommend_products():
    """Recommend products for a customer"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        customer_id = data.get('customer_id')
        if not customer_id:
            return jsonify({"error": "customer_id is required"}), 400
        # Validate customer_id is a number and > 1
        try:
            customer_id_num = int(customer_id)
            if customer_id_num <= 1:
                return jsonify({"error": "customer_id must be greater than 1"}), 400
        except ValueError:
            return jsonify({"error": "customer_id must be a valid integer"}), 400
        
        
        #TODO: change this to a real customer id check
        is_new_customer = int(str(customer_id)) < 1000  # Example
        if is_new_customer:
            # Use loaded popular products if available
            if popular_products is not None:
                top_products = [
                    {"product_id": str(i+1), "name": name, "score": float(sales)}
                    for i, (name, sales) in enumerate(popular_products.items())
                ]
            else:
                top_products = [
                    {"product_id": "P001", "name": "Premium Widget", "score": 0.95},
                    {"product_id": "P002", "name": "Super Gadget", "score": 0.92},
                    {"product_id": "P003", "name": "Ultra Device", "score": 0.89},
                    {"product_id": "P004", "name": "Mega Tool", "score": 0.87},
                    {"product_id": "P005", "name": "Pro Equipment", "score": 0.85}
                ]
        else:
            # Personalized recommendations using SVD model
            if recommendation_model is not None and popular_products is not None:
                try:
                    # Get all product names
                    all_items = list(popular_products.keys())
                    # For demo, recommend top 5 not purchased (simulate)
                    predictions = [
                        (item, recommendation_model.predict(str(customer_id), item).est)
                        for item in all_items
                    ]
                    top_recs = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
                    recommended_products = [
                        {"product_id": str(i+1), "name": item, "score": float(score)}
                        for i, (item, score) in enumerate(top_recs)
                    ]
                except Exception as e:
                    logging.error(f"Error making recommendation prediction: {e}")
                    # Fallback to popular products
                    recommended_products = [
                        {"product_id": "P001", "name": "Premium Widget", "score": 0.95},
                        {"product_id": "P002", "name": "Super Gadget", "score": 0.92},
                        {"product_id": "P003", "name": "Ultra Device", "score": 0.89}
                    ]
            else:
                recommended_products = [
                    {"product_id": "P001", "name": "Premium Widget", "score": 0.95},
                    {"product_id": "P002", "name": "Super Gadget", "score": 0.92},
                    {"product_id": "P003", "name": "Ultra Device", "score": 0.89}
                ]
            top_products = recommended_products
        return jsonify({
            "customer_id": customer_id,
            "is_new_customer": is_new_customer,
            "recommendations": top_products,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500



@app.route('/detect-spam', methods=['POST'])
def detect_spam():
    """Detect spam based on user profile metrics"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        is_valid, error_message = validate_spam_input(data)
        if not is_valid:
            return jsonify({"error": error_message}), 400
        
        # Prepare features for prediction
        if spam_detection_model and spam_scaler:
            # Handle numeric features
            numeric_features = [
                data["Profile_Completeness"],
                data["Customer_Feedback"],
                data["Transaction_History"]
            ]
            
            # Handle categorical features using encoders
            if "Sales_Consistency_encoder" in spam_encoders:
                sales_consistency_encoded = spam_encoders["Sales_Consistency_encoder"].transform([[data["Sales_Consistency"]]])[0]
                numeric_features.append(sales_consistency_encoded)
            else:
                # Fallback: use 0.5 if encoder not available
                numeric_features.append(0.5)
            
            if "Platform_Interaction_encoder" in spam_encoders:
                platform_interaction_encoded = spam_encoders["Platform_Interaction_encoder"].transform([[data["Platform_Interaction"]]])[0]
                numeric_features.append(platform_interaction_encoded)
            else:
                # Fallback: use 0.5 if encoder not available
                numeric_features.append(0.5)
            
            # Scale features
            #TODO: Are Platform_Interaction and Sales_Consistency included here or not?
            features_scaled = spam_scaler.transform([numeric_features])
            
            print("Original features:", numeric_features)
            print("Features scaled:", features_scaled)
            print("Features scaled shape:", features_scaled.shape)
            print("Spam detection model:", spam_detection_model)
            print("Model prediction:", spam_detection_model.predict(features_scaled))
            
            prediction = spam_detection_model.predict(features_scaled)[0]
            
            # Convert prediction to label using Label encoder if available
            if "Label_encoder" in spam_encoders:
                try:
                    label = spam_encoders["Label_encoder"].inverse_transform([prediction])[0]
                    if label == "fake":
                        label = "spam"
                    else:
                        label = "not spam"
                except:
                    # Fallback if inverse transform fails
                    label = "not spam" if prediction == 0 else "spam"
            #else:
            #    # Fallback: map 0/1 to labels
            #    label = "not spam" if prediction == 0 else "spam"
            
            #confidence = float(max(spam_detection_model.predict_proba(features_scaled)[0])) if hasattr(spam_detection_model, 'predict_proba') else None
            
        else:
            # Fallback logic if model or scaler not available

            numeric_features = [
                data["Profile_Completeness"],
                data["Customer_Feedback"],
                data["Transaction_History"]
            ]
            avg_score = sum(numeric_features) / len(numeric_features)
            label = "spam" if avg_score < 0.5 else "not spam"
            confidence = avg_score
        
        return jsonify({
            "input_features": data,
            "prediction": label,
            #"confidence_score": round(confidence, 3) if confidence is not None else None,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in detect-spam endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500



@app.route('/predict-sales', methods=['POST'])
def predict_sales():
    """Predict sales revenue based on input features"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        # Validate input using the validate_sales_input function
        is_valid, error_message = validate_sales_input(data)
        if not is_valid:
            return jsonify({"error": error_message}), 400
        
        if sales_prediction_model is None:
            # Provide fallback prediction using simple heuristics
            logging.warning("Sales prediction model not available, using fallback logic")
            
            # Simple fallback prediction based on input features
            base_revenue = float(data['unit_price']) * int(data['units_sold'])
            
            # Apply some basic multipliers based on features
            multipliers = {
                'product_type': {'electronics': 1.2, 'clothing': 0.9, 'books': 0.8, 'home': 1.1},
                'season': {'summer': 1.1, 'winter': 0.9, 'spring': 1.0, 'fall': 1.05},
                'marketing_channel': {'social_media': 1.15, 'email': 1.0, 'tv': 1.2, 'print': 0.9}
            }
            
            product_mult = multipliers['product_type'].get(data['product_type'], 1.0)
            season_mult = multipliers['season'].get(data['season'], 1.0)
            marketing_mult = multipliers['marketing_channel'].get(data['marketing_channel'], 1.0)
            
            # Apply ad budget effect (simple linear relationship)
            ad_effect = 1.0 + (float(data['ad_budget']) / 10000) * 0.1
            
            predicted_revenue = base_revenue * product_mult * season_mult * marketing_mult * ad_effect
            
            return jsonify({
                'predicted_sales_revenue': round(predicted_revenue, 2),
                'input_features': data,
                'note': 'Using fallback prediction due to model compatibility issues'
            })
        
        # Extract features for model prediction
        features = {
            'product_type': data['product_type'],
            'marketing_channel': data['marketing_channel'],
            'season': data['season'],
            'ad_budget': float(data['ad_budget']),
            'unit_price': float(data['unit_price']),
            'units_sold': int(data['units_sold'])
        }
        
        # Make prediction with the model
        try:
            # Create a DataFrame with the input features
            input_df = pd.DataFrame([features])
            
            # Make prediction (model should handle preprocessing internally)
            prediction = sales_prediction_model.predict(input_df)
            
            # Return prediction
            return jsonify({
                'predicted_sales_revenue': float(prediction[0]),
                'input_features': features
            })
            
        except Exception as e:
            logging.error(f"Error in sales prediction: {str(e)}")
            # Check if it's the XGBoost gpu_id error
            if "gpu_id" in str(e) or "XGBModel" in str(e):
                logging.warning("XGBoost compatibility issue detected during prediction, using fallback logic")
                
                # Use fallback prediction using simple heuristics
                base_revenue = float(data['unit_price']) * int(data['units_sold'])
                
                # Apply some basic multipliers based on features
                multipliers = {
                    'product_type': {'electronics': 1.2, 'clothing': 0.9, 'books': 0.8, 'home': 1.1},
                    'season': {'summer': 1.1, 'winter': 0.9, 'spring': 1.0, 'fall': 1.05},
                    'marketing_channel': {'social_media': 1.15, 'email': 1.0, 'tv': 1.2, 'print': 0.9}
                }
                
                product_mult = multipliers['product_type'].get(data['product_type'], 1.0)
                season_mult = multipliers['season'].get(data['season'], 1.0)
                marketing_mult = multipliers['marketing_channel'].get(data['marketing_channel'], 1.0)
                
                # Apply ad budget effect (simple linear relationship)
                ad_effect = 1.0 + (float(data['ad_budget']) / 10000) * 0.1
                
                predicted_revenue = base_revenue * product_mult * season_mult * marketing_mult * ad_effect
                
                return jsonify({
                    'predicted_sales_revenue': round(predicted_revenue, 2),
                    'input_features': features,
                    'note': 'Using fallback prediction due to XGBoost compatibility issues'
                })
            else:
                return jsonify({'error': f'Prediction error: {str(e)}'}), 500

    except Exception as e:
        logging.error(f"Error in sales prediction endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 