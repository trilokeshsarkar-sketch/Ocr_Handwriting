# Quick Reference: Render Deployment

## Local Testing Before Deployment

```bash
# 1. Check all dependencies
python check_deployment.py

# 2. Run Streamlit locally
streamlit run app.py

# 3. Test with sample images
# Use the UI to test OCR functionality
```

## GitHub Setup (Required for Render)

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Prepare OCR app for Render deployment"

# Push to GitHub
git push origin main
```

## Render Deployment (One-Click via Dashboard)

1. Go to https://render.com/dashboard
2. Click "New +" → "Web Service"
3. Select GitHub repository: `Oce_Custom_Model`
4. Configuration auto-detected from `render.yaml`
5. Click "Deploy"

## Manual Render Configuration

If auto-detection fails, use:
- **Name**: `ocr-streamlit-app`
- **Environment**: Python
- **Region**: Auto-selected
- **Plan**: Pro or Standard
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run app.py --server.port=8501 --server.address=0.0.0.0`

## Environment Variables on Render

Add in Dashboard → Settings → Environment:

```
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
TORCH_HOME=/opt/render/.cache/torch
TRANSFORMERS_CACHE=/opt/render/.cache/transformers
EASYOCR_MODULE_PATH=/opt/render/.cache/easyocr
OCR_OUTPUT_DIR=/tmp/ocr_outputs
PYTHONUNBUFFERED=1
```

## Monitoring

- **Logs**: Dashboard → Logs tab
- **Metrics**: Dashboard → Metrics tab
- **Usage**: Billing section
- **Performance**: Monitor response times

## Resource Recommendations

| Resource | Recommended | Notes |
|----------|------------|-------|
| CPU | 2+ vCPU | PyTorch needs significant CPU |
| Memory | 4GB+ | Model loading + inference |
| Disk | 5GB+ | Model caching |
| Plan | Pro/Standard | Don't use free tier |

## Troubleshooting Commands

```bash
# Check app health
curl https://your-app.onrender.com

# View logs (from Render dashboard)
# Render → Settings → Logs

# Manual restart (from Render dashboard)
# Services → Select service → Restart
```

## Cost Estimation

- **Idle minutes**: First 750 free/month
- **Active**: ~$0.10/hour per CPU
- **Typical cost**: $12-40/month depending on usage

## First Deployment Timeline

- **Initial push**: 2-3 minutes
- **Build phase**: 3-5 minutes (installs dependencies)
- **Model download**: 10-15 minutes (first time only)
- **Total first deployment**: 15-25 minutes
- **Subsequent deploys**: 3-5 minutes (cached models)

## Performance Tips

1. **Upload smaller PDFs** (~5-10MB max)
2. **Limit pages**: Use max_pages setting
3. **Reuse sessions**: App caches pipeline per user session
4. **Monitor cold starts**: Models take time to load
5. **Batch requests**: Process multiple files sequentially

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| Out of memory | Upgrade to higher plan or reduce image size |
| Slow startup | Normal for first deploy, cache on restarts |
| 502 Bad Gateway | Check logs, may need more resources |
| Models not loading | Increase timeout, check disk space |
| CUDA not available | Expected in cloud, uses CPU (slower) |

## Useful Links

- Render Documentation: https://render.com/docs
- Streamlit on Render: https://render.com/docs/deploy-streamlit
- Render Status: https://status.render.com
- Support: https://render.com/support

## Need Help?

1. Check RENDER_DEPLOYMENT.md for detailed guide
2. Review app logs in Render dashboard
3. Run `python check_deployment.py` locally
4. Check GitHub issues in your repo
