# Deployment Guide: Flask AI/ML API on Render

## Prerequisites

- All model files (see below) in the correct folders in your repo
- `requirements.txt` listing all dependencies
- `render.yaml` in the root of your repo
- Your code and models pushed to GitHub

## Required Model Files

- `models/recomndation/svd_collaborative_model.pkl`
- `models/recomndation/popular_products.pkl`
- (Other model files for spam/sales as needed)

## Steps to Deploy on Render

1. Push your code and all model files to GitHub.
2. Ensure `render.yaml` is present in the root directory. Example:
   ```yaml
   services:
     - type: web
       name: flask-ml-api
       runtime: python
       plan: free
       buildCommand: pip install -r requirements.txt
       startCommand: gunicorn app:app
       envVars:
         - key: PYTHON_VERSION
           value: 3.9.16
   ```
3. Go to [Render Dashboard](https://dashboard.render.com/).
4. Click "New +" > "Web Service" and connect your GitHub repo.
5. Render will use `render.yaml` for build/start commands.
6. Wait for the build and deployment to finish. Your API will be live at `https://<your-app-name>.onrender.com`.

## API Response Format

- All endpoints return a `status_code` field in the JSON response.
- JSON key order is not guaranteed (per standard).

## Recommendation Endpoint Logic

- If the customer ID is in the training set, returns personalized recommendations.
- If not, returns top popular products.

## Troubleshooting

- Ensure all model files are present in the repo and in the correct folders.
- Check Render build logs for missing dependencies or files.
- If you get 500 errors, check that all models are loaded correctly.

## API Endpoints

Your deployed API will be available at:

- **Base URL**: `https://your-app-name.onrender.com`
- **Health Check**: `GET /`
- **Models Status**: `GET /models-state`
- **Recommendations**: `POST /recommend`
- **Spam Detection**: `POST /detect-spam`
- **Sales Prediction**: `POST /predict-sales`

## Monitoring and Maintenance

1. **Set up monitoring** in Render dashboard
2. **Configure alerts** for service downtime
3. **Regularly update dependencies**
4. **Monitor API usage and performance**

## Cost Considerations

- **Free Tier**: 750 hours/month, services sleep after inactivity
- **Paid Plans**: Always-on service, more resources, custom domains
- **Bandwidth**: Included in plan limits

## Support

- **Render Documentation**: https://render.com/docs
- **Render Support**: Available in dashboard
- **Community**: Render Discord and forums
