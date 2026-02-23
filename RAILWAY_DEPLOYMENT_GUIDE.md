# 🚂 Railway Deployment Guide - Backend API

## ✅ Files Ready for Deployment

All necessary files have been created and committed:

- ✅ `video_processor_api.py` - Main API server
- ✅ `perfect_hand_signal_detector.py` - Hand signal detection
- ✅ `station_alert_system.py` - Station alert compliance
- ✅ `requirements.txt` - Python dependencies
- ✅ `Procfile` - Railway start command
- ✅ `railway.json` - Railway configuration
- ✅ `.gitignore` - Git ignore rules
- ✅ `Detected_Signals_Lat_Long.xlsx` - Excel rules file
- ✅ `Detected_Signals_Lat_Long_Enhanced.xlsx` - Enhanced Excel rules

## 📤 Step 1: Push to GitHub

### Option A: Use the Script

```bash
cd "/Users/yakesh/Desktop/Railways Project"
chmod +x PUSH_BACKEND_TO_GITHUB.sh
./PUSH_BACKEND_TO_GITHUB.sh
```

### Option B: Manual Commands

```bash
cd "/Users/yakesh/Desktop/Railways Project"

# Update remote to your repository
git remote set-url origin https://github.com/Yakesh70/cvvrs-backend-.git

# Verify remote
git remote -v

# Push to GitHub
git push -u origin main --force
```

**Note**: You may need to authenticate with GitHub (Personal Access Token).

## 🚂 Step 2: Deploy to Railway

1. **Go to Railway**: https://railway.app
2. **Sign in** with GitHub
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose**: `Yakesh70/cvvrs-backend-`
6. **Railway will automatically**:
   - Detect Python
   - Install dependencies from `requirements.txt`
   - Use `Procfile` for start command
   - Deploy your API

## ⚙️ Step 3: Configure Railway

### Environment Variables (Optional)

Railway will automatically set `PORT` environment variable. You can add:

- `CORS_ORIGINS`: Your frontend URL (e.g., `https://cvvrs.vercel.app`)

### Settings

- **Build Command**: Auto-detected (installs from `requirements.txt`)
- **Start Command**: `gunicorn video_processor_api:app --bind 0.0.0.0:$PORT --workers 2 --timeout 300`
- **Health Check**: Railway will check `/api/health` endpoint

## 🔗 Step 4: Get Your API URL

After deployment:

1. Go to your Railway project dashboard
2. Click on your service
3. Go to **Settings** → **Networking**
4. Click **Generate Domain** (or use custom domain)
5. Copy your API URL (e.g., `https://your-app.railway.app`)

## 🔧 Step 5: Update Frontend

1. Go to **Vercel Dashboard** → Your Project (`CVVRS`)
2. Go to **Settings** → **Environment Variables**
3. Add:
   - **Name**: `VITE_API_URL`
   - **Value**: `https://your-app.railway.app` (your Railway API URL)
   - **Environments**: All (Production, Preview, Development)
4. **Redeploy** your frontend

## ✅ Step 6: Test

1. Visit your frontend: https://cvvrs.vercel.app
2. Upload a video
3. Check if processing works!

## 📋 Important Notes

### Model Files

- YOLO model (`yolov8s.pt`) will be **automatically downloaded** on first run
- This may take a few minutes on Railway's first deployment
- Model is ~22MB and will be cached

### Excel Files

- Both Excel files are included in the repository
- The API will use `Detected_Signals_Lat_Long_Enhanced.xlsx` if available
- Falls back to `Detected_Signals_Lat_Long.xlsx` if enhanced version not found

### CORS Configuration

The API already has CORS enabled for all origins. If you want to restrict it:

1. Add `CORS_ORIGINS` environment variable in Railway
2. Update `video_processor_api.py` to use it:
   ```python
   CORS(app, origins=os.environ.get('CORS_ORIGINS', '*').split(','))
   ```

### Timeout Settings

- Railway timeout: 300 seconds (5 minutes)
- For longer videos, you may need to increase this in Railway settings

## 🆘 Troubleshooting

### Build Fails

- Check Railway build logs
- Ensure all dependencies in `requirements.txt` are correct
- Python version should be 3.9+

### API Not Responding

- Check Railway service logs
- Verify `PORT` environment variable is set (Railway sets this automatically)
- Check health endpoint: `https://your-app.railway.app/api/health`

### Model Download Issues

- First deployment may take longer due to model download
- Check Railway logs for download progress
- Model is cached after first download

### CORS Errors

- Verify `CORS_ORIGINS` includes your frontend URL
- Check Railway logs for CORS-related errors

## 📞 Support

If you encounter issues:

1. Check Railway logs: Project → Service → Logs
2. Check Railway build logs: Project → Service → Deployments → Build Logs
3. Verify all files are pushed to GitHub
4. Ensure `requirements.txt` has all dependencies

---

**Ready to deploy!** 🚀
