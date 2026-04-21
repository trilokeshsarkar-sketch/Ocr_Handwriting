# Render Deployment Guide for OCR Pipeline

## Prerequisites
- Render account (https://render.com)
- Git repository with your code pushed to GitHub

## Deployment Steps

### 1. Push Code to GitHub
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Create Render Service
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure settings:
   - **Name**: `ocr-streamlit-app`
   - **Environment**: `Python`
   - **Build Command**: Auto-detected from `render.yaml`
   - **Start Command**: Auto-detected from `render.yaml`
   - **Plan**: Select based on your needs (recommend Pro or higher for this resource-intensive app)

### 3. Set Environment Variables
Add in Render Dashboard → Environment:

```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=false
TORCH_HOME=/opt/render/.cache/torch
TRANSFORMERS_CACHE=/opt/render/.cache/transformers
EASYOCR_MODULE_PATH=/opt/render/.cache/easyocr
OCR_OUTPUT_DIR=/tmp/ocr_outputs
PYTHONUNBUFFERED=1
```

### 4. Resource Optimization

#### For Better Performance:
- **Plan**: Pro or higher (PyTorch models are resource-heavy)
- **Memory**: Minimum 4GB recommended for TrOCR + EasyOCR
- **Disk**: 5GB+ for model caching

#### Cache Strategy:
- Models are cached in `/opt/render/.cache/`
- First deployment will download models (~2GB) - be patient
- Subsequent deployments reuse cached models (faster)

### 5. Monitor Deployment
- Check logs in Render Dashboard
- First startup takes 10-15 minutes (model downloading)
- Subsequent restarts take 2-3 minutes

## Troubleshooting

### Out of Memory Errors
- Consider Render's Standard plan or higher
- Reduce image processing scale in UI
- Implement image size limits

### Slow Startup
- This is normal for first deployment (models download)
- Cache persists between restarts
- Use Render's auto-deploy sparingly

### Model Download Issues
- Check internet connection
- Render's ephemeral disk means models re-download on rebuild
- Consider pre-caching models if deploying frequently

## Performance Tips

1. **Upload smaller images** - Large PDFs can timeout
2. **Limit pages** - Set reasonable max_pages in config
3. **Session reuse** - Streamlit caches pipeline per session
4. **Monitor usage** - Check Render's usage dashboard

## Cost Estimation
- **Pro Plan**: ~$12/month + usage
- **High CPU tasks**: May incur additional compute costs
- **Storage**: Models cached use disk space

## Files for Deployment
- `render.yaml` - Render configuration
- `Procfile` - Alternative process configuration
- `.streamlit/config.toml` - Streamlit optimizations
- `requirements.txt` - Dependencies
