# Flask ML API Deployment Guide - Render

This guide will walk you through deploying your Flask ML API to Render.

## Prerequisites

1. A GitHub account with your Flask API code pushed to a repository
2. A Render account (free tier available)

## Step-by-Step Deployment

### 1. Prepare Your Repository

Make sure your repository contains:

- `app.py` - Your Flask application
- `requirements.txt` - Python dependencies
- `render.yaml` - Render configuration (optional but recommended)
- `models/` - Your ML models directory
- `.dockerignore` - To optimize build process

### 2. Deploy to Render

#### Option A: Using render.yaml (Recommended)

1. **Push your code to GitHub** if you haven't already
2. **Go to [Render Dashboard](https://dashboard.render.com/)**
3. **Click "New +" and select "Blueprint"**
4. **Connect your GitHub repository**
5. **Render will automatically detect the `render.yaml` file**
6. **Click "Apply" to deploy**

#### Option B: Manual Deployment

1. **Go to [Render Dashboard](https://dashboard.render.com/)**
2. **Click "New +" and select "Web Service"**
3. **Connect your GitHub repository**
4. **Configure the service:**
   - **Name**: `flask-ml-api` (or your preferred name)
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free (or your preferred plan)

### 3. Environment Variables (if needed)

If your app requires environment variables:

1. Go to your service dashboard
2. Navigate to "Environment" tab
3. Add any required environment variables

### 4. Model Files

**Important**: Make sure your `models/` directory is included in your repository. Render will download all files during deployment.

### 5. Deployment Verification

After deployment:

1. **Check the deployment logs** for any errors
2. **Test your API endpoints** using the provided URL
3. **Monitor the service** in the Render dashboard

## API Endpoints

Your deployed API will be available at:

- **Base URL**: `https://your-app-name.onrender.com`
- **Health Check**: `GET /`
- **Models Status**: `GET /models-state`
- **Recommendations**: `POST /recommend`
- **Spam Detection**: `POST /detect-spam`
- **Sales Prediction**: `POST /predict-sales`

## Troubleshooting

### Common Issues:

1. **Build Failures**:

   - Check that all dependencies are in `requirements.txt`
   - Ensure Python version compatibility
   - Review build logs for specific errors

2. **Model Loading Issues**:

   - Verify all model files are in the repository
   - Check file paths in your code
   - Ensure models are compatible with the deployed environment

3. **Memory Issues**:

   - Large model files might cause memory problems on free tier
   - Consider upgrading to a paid plan for larger models

4. **Cold Start Delays**:
   - Free tier services sleep after inactivity
   - First request after inactivity may take 30-60 seconds
   - Consider upgrading to paid plan for always-on service

### Performance Optimization:

1. **Model Optimization**:

   - Consider using smaller, optimized models
   - Use model compression techniques
   - Cache frequently used predictions

2. **Code Optimization**:
   - Lazy load models only when needed
   - Implement proper error handling
   - Use async operations where possible

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
