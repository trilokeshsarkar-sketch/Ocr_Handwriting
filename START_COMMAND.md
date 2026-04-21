# Render Start Command Guide

## Current Start Command

```bash
streamlit run app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --logger.level=info \
  --client.toolbarPosition=bottom \
  --client.showErrorDetails=false \
  --client.maxUploadSize=200
```

## Command Breakdown

| Flag | Value | Purpose |
|------|-------|---------|
| `--server.port` | `8501` | Streamlit port (Render default for health checks) |
| `--server.address` | `0.0.0.0` | Listen on all network interfaces |
| `--server.headless` | `true` | Run in headless mode (no browser auto-open) |
| `--server.enableCORS` | `false` | Disable CORS (security) |
| `--logger.level` | `info` | Log level (reduces noise) |
| `--client.toolbarPosition` | `bottom` | UI toolbar at bottom |
| `--client.showErrorDetails` | `false` | Hide error details in production |
| `--client.maxUploadSize` | `200` | Max upload size in MB |

## Environment Variables (Complementary)

```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_LOGGER_LEVEL=info
STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=false
PYTHONUNBUFFERED=1
TORCH_HOME=/opt/render/.cache/torch
TRANSFORMERS_CACHE=/opt/render/.cache/transformers
EASYOCR_MODULE_PATH=/opt/render/.cache/easyocr
OCR_OUTPUT_DIR=/tmp/ocr_outputs
```

## How Render Executes

1. **Build Phase**: Installs dependencies from `requirements.txt`
2. **Start Phase**: Runs the `startCommand`
3. **Health Check**: Pings `http://localhost:8501` every 30 seconds
4. **Service Ready**: When health check passes, app is live

## Troubleshooting Start Command

### App won't start (build succeeds but service crashes)
- Check logs: `tail -f /render/logs/app.log` (in Render dashboard)
- Verify port binding: `ss -tulpn | grep 8501`
- Check memory usage: App requires 4GB+ for models

### Port already in use
- Render automatically assigns 8501
- If conflict, Render escalates and eventually terminates service
- Solution: Higher resource tier

### Health check failing
- App takes time to load models on first start (15+ minutes normal)
- Increase health check timeout in Render settings if available
- Monitor app initialization in logs

### Slow startup
- **Normal**: 15-25 min first deploy (model download)
- **Expected**: 2-5 min subsequent restarts (cached models)
- **Optimize**: Pre-download models in build step if needed

## Production Optimizations

### Add to Start Command for Production:
```bash
# Increase worker threads for concurrent requests
--server.runOnSave=false \
--server.maxUploadSize=200 \
--client.showErrorDetails=false
```

### Consider for Large Deployments:
```bash
# Session persistence
--server.sessionStateTimeout=900
```

## Testing Start Command Locally

```bash
# Test exact Render command locally
streamlit run app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --logger.level=info \
  --client.toolbarPosition=bottom \
  --client.showErrorDetails=false \
  --client.maxUploadSize=200

# Should output something like:
# [date_time] Collecting usage statistics
# [date_time] You can now view your Streamlit app in your browser
# Local URL: http://localhost:8501
```

## Alternative: Using Config File

Instead of command-line flags, you can use `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = false

[logger]
level = "info"

[client]
toolbarPosition = "bottom"
showErrorDetails = false
maxUploadSize = 200
```

Then simple start command:
```bash
streamlit run app.py
```

## Files Using Start Command

- **render.yaml** - Render deployment config
- **Procfile** - Alternative process manager config
- **.streamlit/config.toml** - Streamlit configuration

## Health Check Details

Render monitors your app by:
1. Making HTTP requests to `http://localhost:8501`
2. Checking for successful response (200 OK)
3. If health checks fail 3x in 30 minutes: Service restarts
4. If persistent failures: Service marked as crashed

This is why headless mode and proper port binding are critical.
