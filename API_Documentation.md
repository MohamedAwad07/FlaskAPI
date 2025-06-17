# API Documentation

This document provides detailed information about all API endpoints, including request/response formats, examples, and error handling.

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

| Parameter   | Type   | Required | Description                        |
| ----------- | ------ | -------- | ---------------------------------- |
| customer_id | string | Yes      | Unique identifier for the customer |

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

```json
{
  "error": "customer_id is required"
}
```

```json
{
  "error": "No JSON data provided"
}
```

---

### 3. Spam Detection

**POST** `/detect-spam`

Detect if a user profile is spam based on various metrics.

#### Request Body

```json
{
  "Profile_Completeness": 0.8,
  "Sales_Consistency": 0.6,
  "Customer_Feedback": 0.9,
  "Transaction_History": 0.7,
  "Platform_Interaction": 0.5
}
```

#### Parameters

| Parameter            | Type  | Required | Range | Description                   |
| -------------------- | ----- | -------- | ----- | ----------------------------- |
| Profile_Completeness | float | Yes      | 0-1   | Profile completion percentage |
| Sales_Consistency    | float | Yes      | 0-1   | Consistency of sales behavior |
| Customer_Feedback    | float | Yes      | 0-1   | Customer feedback score       |
| Transaction_History  | float | Yes      | 0-1   | Transaction history quality   |
| Platform_Interaction | float | Yes      | 0-1   | Platform interaction level    |

#### Response

```json
{
  "input_features": {
    "Profile_Completeness": 0.8,
    "Sales_Consistency": 0.6,
    "Customer_Feedback": 0.9,
    "Transaction_History": 0.7,
    "Platform_Interaction": 0.5
  },
  "prediction": "not spam",
  "confidence_score": 0.7,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Response Fields

| Field            | Type   | Description                                 |
| ---------------- | ------ | ------------------------------------------- |
| input_features   | object | The input features used for prediction      |
| prediction       | string | Classification result: "spam" or "not spam" |
| confidence_score | float  | Confidence score of the prediction (0-1)    |

#### Examples

**Not Spam Profile**

```bash
curl -X POST http://localhost:5000/detect-spam \
  -H "Content-Type: application/json" \
  -d '{
    "Profile_Completeness": 0.8,
    "Sales_Consistency": 0.6,
    "Customer_Feedback": 0.9,
    "Transaction_History": 0.7,
    "Platform_Interaction": 0.5
  }'
```

**Spam Profile**

```bash
curl -X POST http://localhost:5000/detect-spam \
  -H "Content-Type: application/json" \
  -d '{
    "Profile_Completeness": 0.2,
    "Sales_Consistency": 0.1,
    "Customer_Feedback": 0.3,
    "Transaction_History": 0.1,
    "Platform_Interaction": 0.2
  }'
```

#### Error Responses

```json
{
  "error": "Missing required field: Profile_Completeness"
}
```

```json
{
  "error": "Field Profile_Completeness must be between 0 and 1"
}
```

---

### 4. Sales Prediction

**POST** `/predict-sales`

Predict sales revenue based on product type, season, and marketing channel.

#### Request Body

```json
{
  "product_type": "electronics",
  "season": "summer",
  "marketing_channel": "social_media"
}
```

#### Parameters

| Parameter         | Type   | Required | Description            | Valid Values                                    |
| ----------------- | ------ | -------- | ---------------------- | ----------------------------------------------- |
| product_type      | string | Yes      | Type of product        | electronics, clothing, books, home, sports      |
| season            | string | Yes      | Season of the year     | spring, summer, fall, winter                    |
| marketing_channel | string | Yes      | Marketing channel used | social_media, email, search, display, affiliate |

#### Response

```json
{
  "input_data": {
    "product_type": "electronics",
    "season": "summer",
    "marketing_channel": "social_media"
  },
  "predicted_revenue": 27300.0,
  "currency": "USD",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Response Fields

| Field             | Type   | Description                        |
| ----------------- | ------ | ---------------------------------- |
| input_data        | object | The input data used for prediction |
| predicted_revenue | float  | Predicted revenue amount           |
| currency          | string | Currency of the prediction         |

#### Examples

**High Revenue Prediction**

```bash
curl -X POST http://localhost:5000/predict-sales \
  -H "Content-Type: application/json" \
  -d '{
    "product_type": "electronics",
    "season": "summer",
    "marketing_channel": "social_media"
  }'
```

**Low Revenue Prediction**

```bash
curl -X POST http://localhost:5000/predict-sales \
  -H "Content-Type: application/json" \
  -d '{
    "product_type": "books",
    "season": "fall",
    "marketing_channel": "display"
  }'
```

#### Error Responses

```json
{
  "error": "Missing required field: product_type"
}
```

```json
{
  "error": "Field product_type cannot be empty"
}
```

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
  "Sales_Consistency": 0.6,
  "Customer_Feedback": 0.9,
  "Transaction_History": 0.7,
  "Platform_Interaction": 0.5
}
```

### Sales Prediction

```
POST {{base_url}}/predict-sales
Headers: Content-Type: application/json
Body (raw JSON):
{
  "product_type": "electronics",
  "season": "summer",
  "marketing_channel": "social_media"
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

## Model Integration

### Current Implementation

The API currently uses placeholder logic for predictions. To integrate your trained models:

1. Save your models as `.pkl` files in the `models/` directory
2. Update the prediction logic in `app.py` to use your models
3. Ensure your models accept the expected input format

### Model Requirements

- **Recommendation Model**: Should accept customer_id and return product recommendations
- **Spam Detection Model**: Should accept 5 float features (0-1) and return binary classification
- **Sales Prediction Model**: Should accept 3 string features and return revenue prediction

### Example Model Integration

```python
# In app.py, replace placeholder logic with:
if recommendation_model:
    recommendations = recommendation_model.predict(customer_id)
    # Process recommendations and return

if spam_detection_model:
    prediction = spam_detection_model.predict([features])[0]
    # Process prediction and return

if sales_prediction_model:
    revenue = sales_prediction_model.predict([features])[0]
    # Process revenue and return
```

---

## Support

For API support or questions:

- Check the health endpoint for model status
- Review error messages for debugging
- Ensure all required fields are provided
- Verify data types and value ranges
