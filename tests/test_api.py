import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"
HEADERS = {"Content-Type": "application/json"}

def test_health_check():
    """Test the health check endpoint"""
    print("Testing Health Check...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Health check passed!")
        else:
            print("❌ Health check failed!")
            
    except Exception as e:
        print(f"❌ Error testing health check: {e}")
    
    print("-" * 50)

def test_recommendation():
    """Test the recommendation endpoint"""
    print("Testing Recommendation Endpoint...")
    
    # Test cases
    test_cases = [
        {"customer_id": "500", "description": "New customer (ID < 1000)"},
        {"customer_id": "12345", "description": "Existing customer (ID >= 1000)"},
        {"customer_id": "99999", "description": "Large customer ID"}
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['description']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/recommend",
                headers=HEADERS,
                json={"customer_id": test_case["customer_id"]}
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                print("✅ Recommendation test passed!")
            else:
                print("❌ Recommendation test failed!")
                
        except Exception as e:
            print(f"❌ Error testing recommendation: {e}")
    
    print("-" * 50)

def test_spam_detection():
    """Test the spam detection endpoint"""
    print("Testing Spam Detection Endpoint...")
    
    # Test cases
    test_cases = [
        {
            "data": {
                "Profile_Completeness": 0.8,
                "Sales_Consistency": 0.6,
                "Customer_Feedback": 0.9,
                "Transaction_History": 0.7,
                "Platform_Interaction": 0.5
            },
            "description": "Not spam profile (high scores)"
        },
        {
            "data": {
                "Profile_Completeness": 0.2,
                "Sales_Consistency": 0.1,
                "Customer_Feedback": 0.3,
                "Transaction_History": 0.1,
                "Platform_Interaction": 0.2
            },
            "description": "Spam profile (low scores)"
        },
        {
            "data": {
                "Profile_Completeness": 0.5,
                "Sales_Consistency": 0.5,
                "Customer_Feedback": 0.5,
                "Transaction_History": 0.5,
                "Platform_Interaction": 0.5
            },
            "description": "Borderline profile (medium scores)"
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['description']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/detect-spam",
                headers=HEADERS,
                json=test_case["data"]
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                print("✅ Spam detection test passed!")
            else:
                print("❌ Spam detection test failed!")
                
        except Exception as e:
            print(f"❌ Error testing spam detection: {e}")
    
    print("-" * 50)

def test_sales_prediction():
    """Test the sales prediction endpoint"""
    print("Testing Sales Prediction Endpoint...")
    
    # Test cases
    test_cases = [
        {
            "data": {
                "product_type": "electronics",
                "season": "summer",
                "marketing_channel": "social_media"
            },
            "description": "High revenue prediction"
        },
        {
            "data": {
                "product_type": "books",
                "season": "fall",
                "marketing_channel": "display"
            },
            "description": "Low revenue prediction"
        },
        {
            "data": {
                "product_type": "clothing",
                "season": "spring",
                "marketing_channel": "email"
            },
            "description": "Medium revenue prediction"
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['description']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict-sales",
                headers=HEADERS,
                json=test_case["data"]
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                print("✅ Sales prediction test passed!")
            else:
                print("❌ Sales prediction test failed!")
                
        except Exception as e:
            print(f"❌ Error testing sales prediction: {e}")
    
    print("-" * 50)

def test_error_handling():
    """Test error handling for invalid inputs"""
    print("Testing Error Handling...")
    
    # Test cases for invalid inputs
    error_test_cases = [
        {
            "endpoint": "/recommend",
            "data": {},
            "description": "Missing customer_id"
        },
        {
            "endpoint": "/detect-spam",
            "data": {
                "Profile_Completeness": 0.8,
                "Sales_Consistency": 0.6
                # Missing required fields
            },
            "description": "Missing required fields"
        },
        {
            "endpoint": "/detect-spam",
            "data": {
                "Profile_Completeness": 1.5,  # Invalid range
                "Sales_Consistency": 0.6,
                "Customer_Feedback": 0.9,
                "Transaction_History": 0.7,
                "Platform_Interaction": 0.5
            },
            "description": "Invalid value range"
        },
        {
            "endpoint": "/predict-sales",
            "data": {
                "product_type": "",
                "season": "summer",
                "marketing_channel": "social_media"
            },
            "description": "Empty string"
        }
    ]
    
    for test_case in error_test_cases:
        print(f"\nTesting: {test_case['description']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}{test_case['endpoint']}",
                headers=HEADERS,
                json=test_case["data"]
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 400:
                print("✅ Error handling test passed!")
            else:
                print("❌ Error handling test failed!")
                
        except Exception as e:
            print(f"❌ Error testing error handling: {e}")
    
    print("-" * 50)

def test_invalid_endpoints():
    """Test invalid endpoints"""
    print("Testing Invalid Endpoints...")
    
    # Test invalid endpoints
    invalid_endpoints = [
        "/invalid",
        "/recommend/invalid",
        "/health/invalid"
    ]
    
    for endpoint in invalid_endpoints:
        print(f"\nTesting invalid endpoint: {endpoint}")
        
        try:
            response = requests.get(f"{BASE_URL}{endpoint}")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 404:
                print("✅ Invalid endpoint test passed!")
            else:
                print("❌ Invalid endpoint test failed!")
                
        except Exception as e:
            print(f"❌ Error testing invalid endpoint: {e}")
    
    print("-" * 50)

def main():
    """Run all tests"""
    print("🚀 Starting API Tests...")
    print("=" * 50)
    
    # Wait a moment for the server to be ready
    time.sleep(1)
    
    # Run all tests
    test_health_check()
    test_recommendation()
    test_spam_detection()
    test_sales_prediction()
    test_error_handling()
    test_invalid_endpoints()
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main() 