# API Documentation

This document provides detailed information about all API endpoints, including request/response formats, examples, error handling, fallback logic, and version compatibility notes.

## Base URL

- **Development**: `http://localhost:5000`
- **Production**: `https://your-domain.com`

## Authentication

Currently, no authentication is required for the API endpoints.

## Response Format

All API responses are in JSON format and include a timestamp:

```json
{
  "data": "...",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error description",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

Common HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found
- `405` - Method Not Allowed
- `500` - Internal Server Error

---

## Version Compatibility

> **Important:** The API and models require the same library versions for both training and inference. If you see warnings about version mismatches, retrain and re-save models using the current environment for production reliability. See `requirements.txt` for the exact versions used.

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check the API status and model loading status.

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "models_loaded": {
    "recommendation": true,
    "spam_detection": true,
    "sales_prediction": true
  }
}
```

#### Example

```bash
curl -X GET http://localhost:5000/health
```

**Status Codes:**

- 200: API is running

---

### 2. Product Recommendation

**POST** `/recommend`

Get product recommendations for a customer based on their ID.

#### Request Body

```json
{
  "customer_id": "string"
}
```

#### Parameters

| Parameter   | Type   | Required | Description                        | Example Value |
| ----------- | ------ | -------- | ---------------------------------- | ------------- |
| customer_id | string | Yes      | Unique identifier for the customer | "12345"       |

#### Response

```json
{
  "customer_id": "12345",
  "is_new_customer": false,
  "recommendations": [
    {
      "product_id": "P101",
      "name": "Personalized Item 1",
      "score": 0.88
    },
    {
      "product_id": "P102",
      "name": "Personalized Item 2",
      "score": 0.85
    }
  ],
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Response Fields

| Field                        | Type    | Description                           |
| ---------------------------- | ------- | ------------------------------------- |
| customer_id                  | string  | The customer ID from the request      |
| is_new_customer              | boolean | Whether this is a new customer        |
| recommendations              | array   | List of recommended products          |
| recommendations[].product_id | string  | Unique product identifier             |
| recommendations[].name       | string  | Product name                          |
| recommendations[].score      | float   | Recommendation confidence score (0-1) |
| timestamp                    | string  | ISO timestamp of the response         |

#### Examples

**New Customer (ID < 1000)**

```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "500"}'
```

**Existing Customer (ID >= 1000)**

```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "12345"}'
```

#### Error Responses

- **Missing Field:**
  ```json
  { "error": "customer_id is required" }
  ```
- **No JSON Data:**
  ```json
  { "error": "No JSON data provided" }
  ```

**Status Codes:**

- 200: Success
- 400: Missing or invalid input

---

### 3. Spam Detection

**POST** `/detect-spam`

Detect if a user profile is spam based on various metrics.

#### Request Body

```json
{
  "Profile_Completeness": 0.8,
  "Sales_Consistency": "high",
  "Customer_Feedback": 0.9,
  "Transaction_History": 0.7,
  "Platform_Interaction": "meduim"
}
```

#### Parameters

| Parameter            | Type   | Required | Description                                                  | Example Values                   |
| -------------------- | ------ | -------- | ------------------------------------------------------------ | -------------------------------- |
| Profile_Completeness | float  | Yes      | Profile completion (0-1). Should be a float between 0 and 1. | 0.8, 1.0, 0.45                   |
| Sales_Consistency    | string | Yes      | Categorical: consistency of sales.                           | "high", "medium", "low"          |
| Customer_Feedback    | float  | Yes      | Customer feedback score (0-1).                               | 0.9, 0.5, 0.0                    |
| Transaction_History  | float  | Yes      | Transaction history quality (0-1).                           | 0.7, 0.2, 1.0                    |
| Platform_Interaction | string | Yes      | Categorical: user's interaction with the platform.           | "active", "inactive", "moderate" |

**If a required field is missing or invalid, the API returns a 400 error with a descriptive message.**

#### Success Response

```json
{
  "input_features": {
    "Profile_Completeness": 0.8,
    "Sales_Consistency": "high",
    "Customer_Feedback": 0.9,
    "Transaction_History": 0.7,
    "Platform_Interaction": "active"
  },
  "prediction": "not spam",
  "confidence_score": 0.95,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Error Responses

- **Missing Field:**
  ```json
  { "error": "Missing required field: Profile_Completeness" }
  ```
- **Invalid Value:**
  ```json
  { "error": "Field Profile_Completeness must be between 0 and 1" }
  ```

#### Fallback Logic

If the spam detection model or scaler is not available, the API uses a simple average of the numeric features to determine if the profile is spam.

**Status Codes:**

- 200: Success
- 400: Missing or invalid input
- 500: Internal server error

---

### 4. Sales Prediction

**POST** `/predict-sales`

Predict sales revenue based on product type, season, marketing channel, ad budget, unit price, and units sold.

#### Request Body

```json
{
  "product_type": "electronics",
  "season": "summer",
  "marketing_channel": "social_media",
  "ad_budget": 5000,
  "unit_price": 100,
  "units_sold": 200
}
```

#### Parameters

| Parameter         | Type   | Required | Description            | Valid Values                                               | Example Values |
| ----------------- | ------ | -------- | ---------------------- | ---------------------------------------------------------- | -------------- |
| product_type      | string | Yes      | Type of product        | electronics, clothing, books, home, sports                 | "electronics"  |
| season            | string | Yes      | Season of the year     | spring, summer, fall, winter                               | "summer"       |
| marketing_channel | string | Yes      | Marketing channel used | social_media, email, tv, print, search, display, affiliate | "social_media" |
| ad_budget         | float  | Yes      | Advertising budget     | Any positive number                                        | 5000           |
| unit_price        | float  | Yes      | Price per unit         | Any positive number                                        | 100            |
| units_sold        | int    | Yes      | Number of units sold   | Any positive integer                                       | 200            |

#### Success Response

```json
{
  "predicted_sales_revenue": 27300.0,
  "input_features": {
    "product_type": "electronics",
    "season": "summer",
    "marketing_channel": "social_media",
    "ad_budget": 5000,
    "unit_price": 100,
    "units_sold": 200
  },
  "note": "Using fallback prediction due to model compatibility issues",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Error Responses

- **Missing Field:**
  ```json
  { "error": "Missing required field: product_type" }
  ```
- **Invalid Value:**
  ```json
  { "error": "Field product_type cannot be empty" }
  ```

#### Fallback Logic

If the model is not available or there are compatibility issues, the API will use fallback logic based on simple heuristics (e.g., multiplying unit price, units sold, and applying multipliers for product type, season, and marketing channel).

**Status Codes:**

- 200: Success
- 400: Missing or invalid input
- 500: Internal server error

---

## Common Error Scenarios

- **Missing required field:**
  - The API returns a 400 error with a message indicating which field is missing.
- **Invalid data type:**
  - The API returns a 400 error with a message indicating the expected type.
- **Model not loaded:**
  - The API uses fallback logic and may include a note in the response.
- **Internal server error:**
  - The API returns a 500 error with a generic message.

---

## Testing with Postman

### Collection Setup

1. Create a new collection in Postman
2. Set the base URL variable: `{{base_url}}` = `http://localhost:5000`
3. Import the following requests:

### Health Check

```
GET {{base_url}}/health
```

### Recommendation

```
POST {{base_url}}/recommend
Headers: Content-Type: application/json
Body (raw JSON):
{
  "customer_id": "12345"
}
```

### Spam Detection

```
POST {{base_url}}/detect-spam
Headers: Content-Type: application/json
Body (raw JSON):
{
  "Profile_Completeness": 0.8,
  "Sales_Consistency": "high",
  "Customer_Feedback": 0.9,
  "Transaction_History": 0.7,
  "Platform_Interaction": "medium"
}
```

### Sales Prediction

```
POST {{base_url}}/predict-sales
Headers: Content-Type: application/json
Body (raw JSON):
{
  "product_type": "Camera",
  "season": "Fall",
  "marketing_channel": "Email",
  "ad_budget": 5000,
  "unit_price": 100,
  "units_sold": 200
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. Consider implementing rate limiting for production use.

## CORS

CORS is enabled for all origins. The API supports:

- GET, POST, OPTIONS methods
- All headers
- All origins

## Model Integration and Compatibility

- All models are loaded from the `models/` directory and subfolders.
- The API is designed to use the exact library versions as used during model training and saving. If you see warnings about version mismatches, retraining and re-saving models with the current environment is recommended for production.
- If a model or preprocessor is not available or fails to load, the API will use fallback logic to provide a best-effort response.

---

## Support

For API support or questions:

- Check the health endpoint for model status
- Review error messages for debugging
- Ensure all required fields are provided
- Verify data types and value ranges
