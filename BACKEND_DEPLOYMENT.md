# Backend Deployment Guide

## Quick Deploy to Render.com (Recommended - FREE)

### Step 1: Prepare Your Repository
```bash
git add .
git commit -m "Prepare backend for deployment"
git push origin main
```

### Step 2: Deploy on Render

1. **Go to [render.com](https://render.com)** and sign up/login with GitHub

2. **Click "New +" → "Web Service"**

3. **Connect Repository**: Select your `canteen-menu-optimizer` repository

4. **Configure Service**:
   - **Name**: `canteen-backend` (or any name you prefer)
   - **Region**: Oregon (US West) or closest to you
   - **Branch**: `main`
   - **Root Directory**: `canteen-App/backend`
   - **Runtime**: `Python 3`
   - **Build Command**: 
     ```bash
     pip install -r requirements.txt && cp ../model/canteen_prediction_model.joblib ./
     ```
   - **Start Command**: 
     ```bash
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```

5. **Environment Variables** (Optional):
   - `PYTHON_VERSION`: `3.11.0`

6. **Click "Create Web Service"**

7. **Wait 2-3 minutes** for deployment to complete

8. **Copy your backend URL** (e.g., `https://canteen-backend.onrender.com`)

### Step 3: Update Frontend Configuration

1. Go to **Streamlit Cloud** → Your App → **Settings** → **Secrets**

2. Add this configuration:
   ```toml
   BACKEND_URL = "https://your-backend-url.onrender.com"
   ```
   Replace `your-backend-url` with your actual Render URL

3. **Save** - Your app will restart automatically

4. **Refresh your Streamlit app** - Demo mode should be gone!

---

## Alternative: Deploy to Railway.app

### Step 1: Deploy on Railway

1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will auto-detect Python and deploy
5. Add environment variable: `PORT=8000`
6. Copy the generated URL

### Step 2: Update Streamlit Secrets (same as above)

---

## Alternative: Deploy to Heroku

### Prerequisites
```bash
# Install Heroku CLI
# Windows: Download from https://devcenter.heroku.com/articles/heroku-cli
# Mac: brew install heroku/brew/heroku
# Linux: curl https://cli-assets.heroku.com/install.sh | sh
```

### Deploy Steps
```bash
# Login to Heroku
heroku login

# Create new app
heroku create canteen-backend

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git subtree push --prefix canteen-App/backend heroku main

# Check logs
heroku logs --tail
```

---

## Troubleshooting

### Issue: Model file not found
**Solution**: Make sure the build command copies the model:
```bash
pip install -r requirements.txt && cp ../model/canteen_prediction_model.joblib ./
```

### Issue: Port binding error
**Solution**: Ensure start command uses `$PORT`:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Issue: CORS errors
**Solution**: Backend already has CORS configured for all origins. If issues persist, check browser console.

### Issue: Deployment timeout
**Solution**: 
- Use Python 3.11 (faster)
- Ensure requirements.txt has version constraints
- Check Render/Railway logs for specific errors

---

## Testing Your Backend

Once deployed, test these endpoints:

1. **Health Check**:
   ```bash
   curl https://your-backend-url.com/health
   ```

2. **Root Endpoint**:
   ```bash
   curl https://your-backend-url.com/
   ```

3. **Prediction** (using curl):
   ```bash
   curl -X POST https://your-backend-url.com/predict \
     -H "Content-Type: application/json" \
     -d '{
       "age": 21,
       "height_cm": 175,
       "weight_kg": 70,
       "spice_tolerance": 7,
       "sweet_tooth_level": 6,
       "eating_out_per_week": 4,
       "food_budget_per_meal": 200,
       "cuisine_top1": "Indian"
     }'
   ```

---

## Cost Considerations

- **Render Free Tier**: 750 hours/month, sleeps after 15 min inactivity
- **Railway Free Tier**: $5 credit/month, ~500 hours
- **Heroku**: No longer has free tier (starts at $7/month)

**Recommendation**: Use Render for free deployment. It will sleep when inactive but wake up automatically when accessed (takes ~30 seconds on first request).

---

## Need Help?

If you encounter issues:
1. Check deployment logs on your platform
2. Verify model file is being copied correctly
3. Test backend endpoints directly before connecting frontend
4. Check Streamlit Cloud logs for connection errors
