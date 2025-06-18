from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import json
from datetime import datetime
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

# Global variables for models and preprocessors
recommendation_model = None
popular_products = None
spam_detection_model = None
spam_scaler = None
spam_encoders = {}
sales_prediction_model = None
sales_preprocessors = {}
sales_product_type_encoder = None
sales_marketing_channel_encoder = None
sales_season_encoder = None
sales_scaler = None

def load_models():
    """Load all ML models from their respective subdirectories"""
    global recommendation_model, popular_products, spam_detection_model, spam_scaler, spam_encoders, sales_prediction_model, sales_preprocessors, sales_product_type_encoder, sales_marketing_channel_encoder, sales_season_encoder, sales_scaler
    try:
        # Recommendation models
        rec_dir = os.path.join("models", "recomndation")
        rec_model_path = os.path.join(rec_dir, "svd_collaborative_model.pkl")
        pop_products_path = os.path.join(rec_dir, "popular_products.pkl")
        try:
            if os.path.exists(rec_model_path):
                recommendation_model = joblib.load(rec_model_path)
                logging.info("Loaded SVD recommendation model.")
                #print('Model type:', type(recommendation_model))
                #print('Model attributes:', dir(recommendation_model))
                #print('Predict method help:')
                #help(recommendation_model.predict)
                #print('Number of latent factors:', getattr(recommendation_model, 'n_factors', 'N/A'))
                #print('Trainset:', getattr(recommendation_model, 'trainset', 'N/A'))
                #print("-------------------------------- user inner ids" + str(list(recommendation_model.trainset.all_users())))
                #print("-------------------------------- item inner ids" + str(list(recommendation_model.trainset.all_items())))
            else:
                logging.warning("SVD recommendation model not found.")
        except ImportError as e:
            if "surprise" in str(e):
                logging.warning("Surprise library not available. Recommendation model will use fallback logic.")
            else:
                logging.error(f"Error loading recommendation model: {e}")
        # Load popular products model
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



        # Sales prediction model
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
            
        # Load sales preprocessors
        if sales_prediction_model:
            sales_preprocessors = {
                'product_type': joblib.load(os.path.join(sales_dir, 'product_type_encoder.pkl')),
                'marketing_channel': joblib.load(os.path.join(sales_dir, 'marketing_channel_encoder.pkl')),
                'season': joblib.load(os.path.join(sales_dir, 'season_encoder.pkl')),
                'scaler': joblib.load(os.path.join(sales_dir, 'sales_scaler.pkl'))
            }
            sales_product_type_encoder = sales_preprocessors['product_type']
            sales_marketing_channel_encoder = sales_preprocessors['marketing_channel']
            sales_season_encoder = sales_preprocessors['season']
            sales_scaler = sales_preprocessors['scaler']
        else:
            logging.warning("Sales prediction model not loaded, sales preprocessors not available")
            
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")

# Load models
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
    
    # Valid categories from the training data
    valid_product_types = ["Camera", "Headphones", "Laptop", "Smartphone", "TV", "Tablet", "Watch"]
    valid_seasons = ["Fall", "Spring", "Summer", "Winter"]
    valid_marketing_channels = ["Affiliate", "Direct", "Email", "Search Engine", "Social Media"]
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        
        # Validate string fields
        if field in ["product_type", "season", "marketing_channel"]:
            if not isinstance(data[field], str):
                return False, f"Field {field} must be a string"
            if not data[field].strip():
                return False, f"Field {field} cannot be empty"
            
            # Check if the value is in the valid categories
            if field == "product_type" and data[field] not in valid_product_types:
                return False, f"Invalid product_type. Must be one of: {valid_product_types}"
            elif field == "season" and data[field] not in valid_seasons:
                return False, f"Invalid season. Must be one of: {valid_seasons}"
            elif field == "marketing_channel" and data[field] not in valid_marketing_channels:
                return False, f"Invalid marketing_channel. Must be one of: {valid_marketing_channels}"
        
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


#End Points

@app.route('/', methods=['GET'])
def welcome():
    """Welcome page with API information"""
    return jsonify({
        "status_code": 200,
        "message": "Welcome to Flask AI/ML API Backend",
        "description": "A REST API serving three machine learning models for recommendation, spam detection, and sales prediction",
        "version": "1.0.0",
        "endpoints": {
            "health": {
                "url": "/health",
                "method": "GET",
                "description": "Check API status and model loading status"
            },
            "recommendation_post": {
                "url": "/recommend",
                "method": "POST",
                "description": "Get product recommendations for customers using POST",
                "example_input": {"customer_id": "12345"}
            },
            "recommendation_get": {
                "url": "/recommend/{customer_id}",
                "method": "GET",
                "description": "Get product recommendations for customers using GET with customer ID in URL",
                "example": "/recommend/12345"
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
                    "product_type": "Laptop",
                    "season": "Summer",
                    "marketing_channel": "Social Media",
                    "ad_budget": 10000,
                    "unit_price": 999.99,
                    "units_sold": 25
                },
                "valid_categories": {
                    "product_type": ["Camera", "Headphones", "Laptop", "Smartphone", "TV", "Tablet", "Watch"],
                    "season": ["Fall", "Spring", "Summer", "Winter"],
                    "marketing_channel": ["Affiliate", "Direct", "Email", "Search Engine", "Social Media"]
                }
            }
        },
        "models_status": {
            "recommendation": recommendation_model is not None,
            "spam_detection": spam_detection_model is not None,
            "sales_prediction": sales_prediction_model is not None
        },
        "documentation": "See /health for detailed model status and API_Documentation.md for complete documentation",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/models-state', methods=['GET'])
def models_check():
    return jsonify({
        "status_code": 200,
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "recommendation": recommendation_model is not None,
            "spam_detection": spam_detection_model is not None,
            "sales_prediction": sales_prediction_model is not None
        },
        "spam_preprocessors_loaded": {
            "Platform_Interaction_encoder": spam_encoders["Platform_Interaction_encoder"] is not None,
            "Sales_Consistency_encoder": spam_encoders["Sales_Consistency_encoder"] is not None,
            "Label_encoder": spam_encoders["Label_encoder"] is not None
        },
        "sales_preprocessors_loaded": {
            "product_type_encoder": sales_product_type_encoder is not None,
            "marketing_channel_encoder": sales_marketing_channel_encoder is not None,
            "season_encoder": sales_season_encoder is not None,
            "sales_scaler": sales_scaler is not None
        },
    }), 200

@app.route('/recommend', methods=['POST'])
def recommend_products():
    """Recommend products for a customer based on customer ID"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status_code": 400, "error": "No JSON data provided"}), 400
        if 'customer_id' not in data:
            return jsonify({"status_code": 400, "error": "customer_id is required in JSON body"}), 400
        customer_id = data.get('customer_id')
        if customer_id is None:
            return jsonify({"status_code": 400, "error": "customer_id cannot be null"}), 400
        if customer_id == "":
            return jsonify({"status_code": 400, "error": "customer_id cannot be empty"}), 400
        if not isinstance(customer_id, int):
            return jsonify({"status_code": 400, "error": "customer_id must be an integer, not a string or other type"}), 400
        if customer_id <= 0:
            return jsonify({"status_code": 400, "error": "customer_id must be greater than 0"}), 400
        
        
        if recommendation_model is not None:
            try:
                logger.info(f"Making recommendations for customer ID: {customer_id}")
                trainset = recommendation_model.trainset
                if customer_id in trainset.all_users():
                    # Existing user: personalized recommendations
                    all_items = trainset.all_items()
                    predictions = []
                    for item_id in all_items:
                        try:
                            item_name = f"Item_{item_id}"
                            prediction = recommendation_model.predict(customer_id, item_id)
                            predictions.append({
                                "item_id": item_id,
                                "item_name": item_name,
                                "predicted_rating": prediction.est,
                                "confidence": prediction.details.get('was_impossible', False)
                            })
                        except Exception as e:
                            logger.warning(f"Could not predict for item {item_id}: {e}")
                            continue
                    predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
                    top_recommendations = predictions[:10]
                    logger.info(f"Generated {len(top_recommendations)} personalized recommendations")
                    return jsonify({
                        "status_code": 200,
                        "customer_id": customer_id,
                        "user_type": "existing",
                        "recommendations": top_recommendations,
                        "model_used": "SVD Collaborative Filtering",
                        "timestamp": datetime.now().isoformat()
                    }), 200
                else:
                    # New user: return top products from popular_products model
                    if popular_products is not None:
                        top_products = [
                            {"item_id": i+1, "item_name": name, "score": float(score)}
                            for i, (name, score) in enumerate(list(popular_products.items())[:5])
                        ]
                    else:
                        top_products = []
                    logger.info("New user detected. Returning popular products from model.")
                    return jsonify({
                        "status_code": 200,
                        "customer_id": customer_id,
                        "user_type": "new",
                        "recommendations": top_products,
                        "model_used": "Popular Products Model",
                        "timestamp": datetime.now().isoformat()
                    }), 200
            except Exception as e:
                logger.error(f"Error making recommendation prediction: {e}")
                return jsonify({
                    "status_code": 500,
                    "error": "Failed to generate recommendations",
                    "customer_id": customer_id,
                    "details": str(e)
                }), 500
        else:
            logger.warning("Recommendation model not available")
            return jsonify({
                "status_code": 500,
                "error": "Recommendation model not loaded",
                "customer_id": customer_id
            }), 500
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {str(e)}")
        return jsonify({"status_code": 500, "error": "Internal server error"}), 500

@app.route('/detect-spam', methods=['POST'])
def detect_spam():
    """Detect spam based on user profile metrics"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status_code": 400, "error": "No JSON data provided"}), 400
        is_valid, error_message = validate_spam_input(data)
        if not is_valid:
            return jsonify({"status_code": 400, "error": error_message}), 400
        
        # Prepare features for prediction
        if spam_detection_model and spam_scaler:

            features = []
            
            # 1. Profile_Completeness (numeric)
            features.append(data["Profile_Completeness"])
            
            # 2. Sales_Consistency (categorical - needs encoding)
            if "Sales_Consistency_encoder" in spam_encoders:
                sales_consistency_encoded = spam_encoders["Sales_Consistency_encoder"].transform([[data["Sales_Consistency"]]])[0][0]
                features.append(sales_consistency_encoded)
            else:
                # Fallback: use 0.5 if encoder not available
                features.append(0.5)
            
            features.append(data["Customer_Feedback"])
            
            features.append(data["Transaction_History"])
            
            if "Platform_Interaction_encoder" in spam_encoders:
                platform_interaction_encoded = spam_encoders["Platform_Interaction_encoder"].transform([[data["Platform_Interaction"]]])[0][0]
                features.append(platform_interaction_encoded)
            else:
                # Fallback: use 0.5 if encoder not available
                features.append(0.5)
            
            logger.info(f"Encoded features before scaling: {features}")
            
            features_scaled = spam_scaler.transform([features])
            
            logger.info(f"Scaled features: {features_scaled}")
            logger.info(f"Scaled features shape: {features_scaled.shape}")
            
            # Model prediction
            prediction = spam_detection_model.predict([features])[0]
            logger.info(f"Raw model prediction: {prediction}")
            
            # Prediction probabilities (if available)
            if hasattr(spam_detection_model, 'predict_proba'):
                proba = spam_detection_model.predict_proba([features])[0]
                logger.info(f"Prediction probabilities: {proba}")
            else:
                proba = None
            
            # Convert prediction to label using Label encoder if available
            if "Label_encoder" in spam_encoders:
                try:
                    label = spam_encoders["Label_encoder"].inverse_transform([prediction])[0]
                    if label == "fake":
                        label = "spam"
                    else:
                        label = "not spam"
                except Exception as e:
                    logger.error(f"Error in label inverse transform: {e}")
                    label = "not spam" if prediction == 0 else "spam"
            #else:
            #    # Fallback: map 0/1 to labels
            #    label = "not spam" if prediction == 0 else "spam"
            
        
        return jsonify({
            "status_code": 200,
            "input_features": data,
            "prediction": label,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error in detect-spam endpoint: {str(e)}")
        return jsonify({"status_code": 500, "error": "Internal server error"}), 500

@app.route('/predict-sales', methods=['POST'])
def predict_sales():
    """Predict sales revenue based on input features"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"status_code": 400, "error": "No JSON data provided"}), 400
            
        is_valid, error_message = validate_sales_input(data)
        if not is_valid:
            return jsonify({"status_code": 400, "error": error_message}), 400
        
        # Check if encoders and scaler are loaded
        if not (sales_product_type_encoder and sales_marketing_channel_encoder and 
                sales_season_encoder and sales_scaler):
            return jsonify({"status_code": 500, "error": "Sales encoders or scaler not loaded"}), 500
        
        # Transform input using encoders and scaler
        try:
            # Encode categorical features
            product_type_encoded = sales_product_type_encoder.transform([data['product_type']])[0]
            season_encoded = sales_season_encoder.transform([data['season']])[0]
            
            # One-hot encode marketing_channel (to match the model's expected features)
            marketing_channel = data['marketing_channel']
            marketing_channel_Affiliate = 1 if marketing_channel == "Affiliate" else 0
            marketing_channel_Direct = 1 if marketing_channel == "Direct" else 0
            marketing_channel_Email = 1 if marketing_channel == "Email" else 0
            marketing_channel_Search_Engine = 1 if marketing_channel == "Search Engine" else 0
            marketing_channel_Social_Media = 1 if marketing_channel == "Social Media" else 0
            
            # Scale numerical features
            numerical_data = np.array([[data['ad_budget'], data['unit_price'], data['units_sold']]])
            numerical_scaled = sales_scaler.transform(numerical_data)[0]
            
            
            #TODO: Check if this is the correct order
            features = [
                numerical_scaled[0],  # ad_budget
                numerical_scaled[1],  # unit_price
                numerical_scaled[2] , # units_sold
                float(product_type_encoded),  # product_type_encoded
                season_encoded,  # season_encoded
                marketing_channel_Affiliate,  # marketing_channel_Affiliate
                marketing_channel_Direct,  # marketing_channel_Direct
                marketing_channel_Email,  # marketing_channel_Email
                marketing_channel_Search_Engine,  # marketing_channel_Search Engine
                marketing_channel_Social_Media,  # marketing_channel_Social Media
            ]
            
            features_array = np.array([features])
            
        except Exception as e:
            logging.error(f"Error transforming features: {str(e)}")
            return jsonify({"status_code": 500, "error": f"Feature transformation error: {str(e)}"}), 500
        
        # Check if model is available
        if sales_prediction_model is None:
            # Provide fallback prediction using simple heuristics
            logging.warning("Sales prediction model not available, using fallback logic")
            
            base_revenue = float(data['unit_price']) * int(data['units_sold'])
            
            # Apply some basic multipliers based on features
            multipliers = {
                'product_type': {'Camera': 1.1, 'Headphones': 0.9, 'Laptop': 1.3, 'Smartphone': 1.2, 'TV': 1.4, 'Tablet': 1.1, 'Watch': 0.8},
                'season': {'Summer': 1.1, 'Winter': 0.9, 'Spring': 1.0, 'Fall': 1.05},
                'marketing_channel': {'Social Media': 1.15, 'Email': 1.0, 'Search Engine': 1.1, 'Affiliate': 0.95, 'Direct': 1.05}
            }
            
            product_mult = multipliers['product_type'].get(data['product_type'], 1.0)
            season_mult = multipliers['season'].get(data['season'], 1.0)
            marketing_mult = multipliers['marketing_channel'].get(data['marketing_channel'], 1.0)
            
            # Apply ad budget effect (simple linear relationship)
            ad_effect = 1.0 + (float(data['ad_budget']) / 10000) * 0.1
            
            predicted_revenue = base_revenue * product_mult * season_mult * marketing_mult * ad_effect
            
            # Convert NumPy types to Python native types for JSON serialization
            features_python = [float(f) if isinstance(f, (np.integer, np.floating)) else int(f) for f in features]
            
            return jsonify({
                "status_code": 200,
                'predicted_sales_revenue': round(predicted_revenue, 2),
                'input_features': data,
                'transformed_features': features_python,
                'note': 'Using fallback prediction due to model feature mismatch'
            }), 200
        
        # Make prediction with the model
        try:
            # Check if the model expects the same number of features as our preprocessing
            if hasattr(sales_prediction_model, 'n_features_in_') and sales_prediction_model.n_features_in_ != len(features):
                logging.warning(f"Model expects {sales_prediction_model.n_features_in_} features but preprocessing provides {len(features)} features. Using fallback logic.")
                raise ValueError("Feature count mismatch")
            
            prediction = sales_prediction_model.predict(features_array)
            
            # Convert NumPy types to Python native types for JSON serialization
            features_python = [float(f) if isinstance(f, (np.integer, np.floating)) else int(f) for f in features]
            
            return jsonify({
                "status_code": 200,
                'predicted_sales_revenue': float(prediction[0]),
                'input_features': data,
            }), 200
            
        except Exception as e:
            logging.error(f"Error in sales prediction: {str(e)}")
            return jsonify({"status_code": 500, "error": f'Prediction error: {str(e)}'}), 500

    except Exception as e:
        logging.error(f"Error in sales prediction endpoint: {str(e)}")
        return jsonify({"status_code": 500, "error": 'Internal server error'}), 500



#End Points for error handling

@app.errorhandler(404)
def not_found(error):
    return jsonify({"status_code": 404, "error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"status_code": 405, "error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status_code": 500, "error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 