from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
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

def load_models():
    """Load all ML models from their respective subdirectories"""
    global recommendation_model, popular_products, spam_detection_model, spam_scaler, spam_encoders, sales_prediction_model
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
        scaler_path = os.path.join(spam_dir, "scaler.pkl")
        
        if os.path.exists(spam_model_path):
            try:
                spam_detection_model = joblib.load(spam_model_path)
                logging.info("Loaded spam detection model.")
            except Exception as e:
                logging.error(f"Error loading spam model: {e}")
        else:
            logging.warning("Spam detection model not found.")
            
        if os.path.exists(scaler_path):
            try:
                spam_scaler = joblib.load(scaler_path)
                logging.info("Loaded spam scaler.")
            except Exception as e:
                logging.error(f"Error loading spam scaler: {e}")
        else:
            logging.warning("Spam scaler not found.")
            
        # Encoders (if needed)
        encoders = ["Platform_Interaction_encoder.pkl", "Sales_Consistency.pkl", "Label.pkl"]
        for enc in encoders:
            enc_path = os.path.join(spam_dir, enc)
            if os.path.exists(enc_path):
                try:
                    spam_encoders[enc] = joblib.load(enc_path)
                    logging.info(f"Loaded spam encoder: {enc}")
                except Exception as e:
                    logging.error(f"Error loading encoder {enc}: {e}")

        # Sales prediction model
        sales_dir = os.path.join("models", "sales")
        sales_model_path = os.path.join(sales_dir, "sales.pkl")
        if os.path.exists(sales_model_path):
            try:
                with open(sales_model_path, "rb") as f:
                    sales_prediction_model = joblib.load(f)
                logging.info("Loaded sales prediction model.")
            except Exception as e:
                logging.error(f"Error loading sales model: {e}")
        else:
            logging.warning("Sales prediction model not found.")
            
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")

def validate_spam_input(data):
    """Validate input for spam detection"""
    required_fields = [
        "Profile_Completeness", "Sales_Consistency", "Customer_Feedback",
        "Transaction_History", "Platform_Interaction"
    ]
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        if not isinstance(data[field], (int, float)):
            return False, f"Field {field} must be a number"
        if data[field] < 0 or data[field] > 1:
            return False, f"Field {field} must be between 0 and 1"
    
    return True, "Valid input"

def validate_sales_input(data):
    """Validate input for sales prediction"""
    required_fields = ["product_type", "season", "marketing_channel"]
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        if not isinstance(data[field], str):
            return False, f"Field {field} must be a string"
        if not data[field].strip():
            return False, f"Field {field} cannot be empty"
    
    return True, "Valid input"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "recommendation": recommendation_model is not None,
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
        is_new_customer = int(str(customer_id)) < 1000  # Example logic
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
        features = [
            data["Profile_Completeness"],
            data["Sales_Consistency"],
            data["Customer_Feedback"],
            data["Transaction_History"],
            data["Platform_Interaction"]
        ]
        if spam_detection_model and spam_scaler:
            features_scaled = spam_scaler.transform([features])
            prediction = spam_detection_model.predict(features_scaled)[0]
            # If model outputs 0/1, map to label
            label = "not spam" if prediction == 0 else "spam"
            confidence = float(max(spam_detection_model.predict_proba(features_scaled)[0])) if hasattr(spam_detection_model, 'predict_proba') else None
        else:
            avg_score = sum(features) / len(features)
            label = "spam" if avg_score < 0.5 else "not spam"
            confidence = avg_score
        return jsonify({
            "input_features": data,
            "prediction": label,
            "confidence_score": round(confidence, 3) if confidence is not None else None,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in detect-spam endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/predict-sales', methods=['POST'])
def predict_sales():
    """Predict sales revenue based on product and marketing data"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        is_valid, error_message = validate_sales_input(data)
        if not is_valid:
            return jsonify({"error": error_message}), 400
        product_type = data["product_type"]
        season = data["season"]
        marketing_channel = data["marketing_channel"]
        if sales_prediction_model:
            # You may need to encode/capitalize/one-hot as per your model's training
            # For demo, just pass as a list (update as needed for your model)
            features = [[product_type, season, marketing_channel]]
            try:
                predicted_revenue = float(sales_prediction_model.predict(features)[0])
            except Exception as e:
                logger.error(f"Sales model prediction error: {e}")
                predicted_revenue = None
        else:
            predicted_revenue = 12000.0
        return jsonify({
            "input_data": data,
            "predicted_revenue": round(predicted_revenue, 2) if predicted_revenue is not None else None,
            "currency": "USD",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in predict-sales endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

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
    # Load models on startup
    load_models()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 