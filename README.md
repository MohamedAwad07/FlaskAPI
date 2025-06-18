# Flask AI/ML API Backend

A Flask-based REST API backend that serves three different machine learning models for recommendation, spam detection, and sales prediction.

## 🚀 Features

- **Three ML Model Endpoints**: Recommendation, Spam Detection, and Sales Prediction
- **CORS Enabled**: Cross-origin requests supported for web and mobile clients
- **Input Validation**: Comprehensive validation for all endpoints
- **Error Handling**: Descriptive error messages and proper HTTP status codes
- **Health Check**: Built-in health monitoring endpoint
- **Model Loading**: Automatic loading of trained models from pickle files
- **Fallback Logic**: Placeholder responses when models are not available

## 📁 Project Structure

```
FlaskAPI/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── API_Documentation.md  # Detailed API documentation
├── models/               # Directory for trained models
│   ├── README.md        # Model placement instructions
│   ├── recommendation_model.pkl
│   ├── spam_detection_model.pkl
│   └── sales_prediction_model.pkl
└── tests/               # Test files (optional)
    └── test_api.py
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**

   ```bash
   git clone <your-repo-url>
   cd FlaskAPI
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your trained models**

   - Place your `.pkl` model files in the `models/` directory
   - See `models/README.md` for naming conventions

5. **Run the application**
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`

## 🔌 API Endpoints

### Health Check

- **URL**: `GET /health`
- **Description**: Check API status and model loading status
- **Response**: JSON with health status and model information

### Product Recommendation

- **URL**: `POST /recommend`
- **Input**: `{"customer_id": "12345"}`
- **Output**: List of recommended products with scores

### Spam Detection

- **URL**: `POST /detect-spam`
- **Input**: JSON with 5 profile metrics (0-1 scale)
- **Output**: Spam classification with confidence score

### Sales Prediction

- **URL**: `POST /predict-sales`
- **Input**: JSON with product type, season, and marketing channel
- **Output**: Predicted revenue value

## 📖 Detailed API Documentation

See `API_Documentation.md` for complete endpoint documentation with examples.

## 🧪 Testing

### Using Postman

1. **Health Check**

   ```
   GET http://localhost:5000/health
   ```

2. **Recommendation**

   ```
   POST http://localhost:5000/recommend
   Content-Type: application/json

   {
     "customer_id": "12345"
   }
   ```

3. **Spam Detection**

   ```
   POST http://localhost:5000/detect-spam
   Content-Type: application/json

   {
     "Profile_Completeness": 0.8,
     "Sales_Consistency": 0.6,
     "Customer_Feedback": 0.9,
     "Transaction_History": 0.7,
     "Platform_Interaction": 0.5
   }
   ```

4. **Sales Prediction**

   ```
   POST http://localhost:5000/predict-sales
   Content-Type: application/json

   {
     "product_type": "electronics",
     "season": "summer",
     "marketing_channel": "social_media"
   }
   ```

### Using curl

```bash
# Health check
curl -X GET http://localhost:5000/health

# Recommendation
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "12345"}'

# Spam detection
curl -X POST http://localhost:5000/detect-spam \
  -H "Content-Type: application/json" \
  -d '{"Profile_Completeness": 0.8, "Sales_Consistency": 0.6, "Customer_Feedback": 0.9, "Transaction_History": 0.7, "Platform_Interaction": 0.5}'

# Sales prediction
curl -X POST http://localhost:5000/predict-sales \
  -H "Content-Type: application/json" \
  -d '{"product_type": "electronics", "season": "summer", "marketing_channel": "social_media"}'
```

## 🚀 Deployment

### Local Development

```bash
python app.py
```

### Production Deployment

#### Using Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Environment Variables

- `PORT`: Port number (default: 5000)
- `FLASK_ENV`: Environment mode (development/production)

### Cloud Deployment

#### Render.com

1. Ensure your repo contains:
   - All code
   - All model files in the correct folders
   - `requirements.txt`
   - `render.yaml`
2. Push to GitHub.
3. Go to [Render](https://dashboard.render.com/), create a new Web Service, and connect your repo.
4. Render will use `render.yaml` for build/start commands.
5. Your API will be live at `https://<your-app-name>.onrender.com`

#### Heroku

1. Create `Procfile`:
   ```
   web: gunicorn app:app
   ```
2. Deploy using Heroku CLI or GitHub integration

#### AWS/DigitalOcean

- Use Docker or direct deployment with gunicorn
- Set up reverse proxy with nginx
- Configure SSL certificates

## 🔧 Configuration

### Model Loading

- Models are automatically loaded from the `models/` directory
- If models are not found, the API uses placeholder logic
- Check the `/health` endpoint to verify model loading status

### CORS Configuration

- CORS is enabled for all origins by default
- Modify in `app.py` if you need specific origin restrictions

### Logging

- Logging is configured to INFO level
- Check console output for model loading and error messages

## 🐛 Troubleshooting

### Common Issues

1. **Models not loading**

   - Check if `.pkl` files exist in `models/` directory
   - Verify file permissions
   - Check console logs for error messages

2. **CORS errors**

   - Ensure Flask-CORS is installed
   - Check if CORS is properly configured in `app.py`

3. **Port already in use**

   - Change port in `app.py` or kill existing process
   - Use `lsof -i :5000` to find process using port 5000

4. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

### Debug Mode

For development, you can enable debug mode:

```python
app.run(host='0.0.0.0', port=port, debug=True)
```

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues:

- Create an issue in the repository
- Check the troubleshooting section above
- Review the API documentation

## 📝 Notes

- All endpoints return a `status_code` in the JSON response.
- No guarantee of key order in JSON (per standard).
- See DEPLOYMENT_GUIDE.md for full deployment instructions.
