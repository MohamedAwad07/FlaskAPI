#!/usr/bin/env python3
"""
Test script to run load_models function from app.py
"""

import sys
import os
import logging

# Configure logging to see the output
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_load_models():
    """Test the load_models function from app.py"""
    try:
        # Import the load_models function from app.py
        from app import load_models, recommendation_model, popular_products, spam_detection_model, spam_scaler, spam_encoders, sales_prediction_model, sales_preprocessors
        
        print("=" * 60)
        print("TESTING MODEL LOADING FUNCTION")
        print("=" * 60)
        
        # Run the load_models function
        print("\nCalling load_models()...")
        load_models()
        
        # Check what models were loaded
        print("\n" + "=" * 40)
        print("MODEL LOADING RESULTS:")
        print("=" * 40)
        
        # Recommendation models
        print(f"\nRecommendation Model: {'✓ Loaded' if recommendation_model is not None else '✗ Not loaded'}")
        print(f"Popular Products: {'✓ Loaded' if popular_products is not None else '✗ Not loaded'}")
        
        # Spam detection models
        print(f"\nSpam Detection Model: {'✓ Loaded' if spam_detection_model is not None else '✗ Not loaded'}")
        print(f"Spam Scaler: {'✓ Loaded' if spam_scaler is not None else '✗ Not loaded'}")
        print(f"Spam Encoders: {len(spam_encoders)} loaded")
        for encoder_name in spam_encoders.keys():
            print(f"  - {encoder_name}")
        
        # Sales prediction models
        print(f"\nSales Prediction Model: {'✓ Loaded' if sales_prediction_model is not None else '✗ Not loaded'}")
        print(f"Sales Preprocessors: {len(sales_preprocessors)} loaded")
        for preprocessor_name in sales_preprocessors.keys():
            print(f"  - {preprocessor_name}")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this script from the same directory as app.py")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load_models() 