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

### 1. Models state Check

**GET** `/models-state`

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
  },
  "sales_preprocessors_loaded": {
    "product_type_encoder": true,
    "marketing_channel_encoder": true,
    "season_encoder": true,
    "sales_scaler": true
  }
}
```

#### Example

```bash
curl -X GET http://localhost:5000/models-state
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
  "Platform_Interaction": "medium"
}
```

#### Parameters

| Parameter            | Type   | Required | Description                                                  | Example Values          |
| -------------------- | ------ | -------- | ------------------------------------------------------------ | ----------------------- |
| Profile_Completeness | float  | Yes      | Profile completion (0-1). Should be a float between 0 and 1. | 0.8, 1.0, 0.45          |
| Sales_Consistency    | string | Yes      | Categorical: consistency of sales.                           | "high", "medium", "low" |
| Customer_Feedback    | float  | Yes      | Customer feedback score (0-1).                               | 0.9, 0.5, 0.0           |
| Transaction_History  | float  | Yes      | Transaction history quality (0-1).                           | 0.7, 0.2, 1.0           |
| Platform_Interaction | string | Yes      | Categorical: user's interaction with the platform.           | "high", "medium", "low" |

**If a required field is missing or invalid, the API returns a 400 error with a descriptive message.**

#### Success Response

```json
{
  "input_features": {
    "Profile_Completeness": 0.8,
    "Sales_Consistency": "high",
    "Customer_Feedback": 0.9,
    "Transaction_History": 0.7,
    "Platform_Interaction": "medium"
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
  "product_type": "Camera",
  "season": "Fall",
  "marketing_channel": "Email",
  "ad_budget": 5000.0,
  "unit_price": 100,
  "units_sold": 15
}
```

#### Parameters

| Parameter         | Type   | Required | Description            | Valid Values                                              | Example Values |
| ----------------- | ------ | -------- | ---------------------- | --------------------------------------------------------- | -------------- |
| product_type      | string | Yes      | Type of product        | Camera, Headphones, Laptop, Smartphone, TV, Tablet, Watch | "Camera"       |
| season            | string | Yes      | Season of the year     | Fall, Spring, Summer, Winter                              | "Fall"         |
| marketing_channel | string | Yes      | Marketing channel used | Affiliate, Direct, Email, Search Engine, Social Media     | "Email"        |
| ad_budget         | float  | Yes      | Advertising budget     | Any positive number                                       | 5000.0         |
| unit_price        | float  | Yes      | Price per unit         | Any positive number                                       | 100            |
| units_sold        | int    | Yes      | Number of units sold   | Any positive integer                                      | 15             |

#### Success Response

```json
{
  "predicted_sales_revenue": 12345.67,
  "input_features": {
    "product_type": "Camera",
    "season": "Fall",
    "marketing_channel": "Email",
    "ad_budget": 5000.0,
    "unit_price": 100,
    "units_sold": 15
  },
  "transformed_features": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
  "feature_names": [
    "ad_budget",
    "unit_price",
    "units_sold",
    "product_type_encoded",
    "season_encoded",
    "marketing_channel_Affiliate",
    "marketing_channel_Direct",
    "marketing_channel_Email",
    "marketing_channel_Search_Engine",
    "marketing_channel_Social_Media"
  ],
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Response Fields

| Field                   | Type   | Description                             |
| ----------------------- | ------ | --------------------------------------- |
| predicted_sales_revenue | float  | Predicted sales revenue amount          |
| input_features          | object | Original input data provided            |
| transformed_features    | array  | Preprocessed features used by the model |
| feature_names           | array  | Names of the transformed features       |
| timestamp               | string | ISO timestamp of the response           |

#### Error Responses

- **Missing Field:**
  ```json
  { "error": "Missing required field: product_type" }
  ```
- **Invalid Category:**
  ```json
  {
    "error": "Invalid product_type. Must be one of: ['Camera', 'Headphones', 'Laptop', 'Smartphone', 'TV', 'Tablet', 'Watch']"
  }
  ```
- **Invalid Number:**
  ```json
  { "error": "Field ad_budget must be a valid number" }
  ```

#### Fallback Logic

If the model is not available or there are compatibility issues, the API will use fallback logic based on simple heuristics:

- Base revenue = unit_price × units_sold
- Apply multipliers for product type, season, and marketing channel
- Apply ad budget effect (simple linear relationship)

The response will include a note indicating fallback usage.

#### Preprocessing Pipeline

The sales prediction endpoint uses the following preprocessing pipeline:

1. **Categorical Encoding:**

   - `product_type`: Label encoded (0-6)
   - `season`: Label encoded (0-3)
   - `marketing_channel`: One-hot encoded (5 binary features)

2. **Numerical Scaling:**

   - `ad_budget`, `unit_price`, `units_sold`: Standardized using fitted scaler

3. **Feature Order:**
   The model expects features in this exact order:
   - `ad_budget` (scaled)
   - `unit_price` (scaled)
   - `units_sold` (scaled)
   - `product_type_encoded` (label encoded)
   - `season_encoded` (label encoded)
   - `marketing_channel_Affiliate` (one-hot)
   - `marketing_channel_Direct` (one-hot)
   - `marketing_channel_Email` (one-hot)
   - `marketing_channel_Search_Engine` (one-hot)
   - `marketing_channel_Social_Media` (one-hot)

The response includes `transformed_features` and `feature_names` for debugging and transparency.

#### Examples

**Valid Request:**

```bash
curl -X POST http://localhost:5000/predict-sales \
  -H "Content-Type: application/json" \
  -d '{
    "product_type": "Laptop",
    "season": "Summer",
    "marketing_channel": "Social Media",
    "ad_budget": 10000,
    "unit_price": 999.99,
    "units_sold": 25
  }'
```

**Invalid Category:**

```bash
curl -X POST http://localhost:5000/predict-sales \
  -H "Content-Type: application/json" \
  -d '{
    "product_type": "InvalidProduct",
    "season": "Summer",
    "marketing_channel": "Social Media",
    "ad_budget": 10000,
    "unit_price": 999.99,
    "units_sold": 25
  }'
```

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

### Models state Check

```
GET {{base_url}}/models-state
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
  "ad_budget": 5000.0,
  "unit_price": 100,
  "units_sold": 15
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
