# Models Directory

Place your trained machine learning models in this directory with the following naming convention:

- `recommendation_model.pkl` - User recommendation model
- `spam_detection_model.pkl` - Spam detection model
- `sales_prediction_model.pkl` - Sales prediction model

## Model Requirements

### Recommendation Model

- Should accept customer_id as input
- Return list of recommended product IDs or names
- Format: Pickle file (.pkl)

### Spam Detection Model

- Should accept 5 features: Profile_Completeness, Sales_Consistency, Customer_Feedback, Transaction_History, Platform_Interaction
- All features should be float values between 0 and 1
- Return binary classification: "spam" or "not spam"
- Format: Pickle file (.pkl)

### Sales Prediction Model

- Should accept 3 features: product_type, season, marketing_channel
- Return predicted revenue value (float)
- Format: Pickle file (.pkl)

## Example Model Loading

The Flask app will automatically load these models on startup. If a model file is not found, the API will use placeholder logic for demonstration purposes.

## Training Your Models

You can train your models using scikit-learn and save them using joblib:

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'models/your_model_name.pkl')
```
