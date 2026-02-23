# Railway Deployment Notes

## ✅ What Railway Needs (You Already Have!)

1. **Procfile** ✓
   - Tells Railway how to start your app
   - Your file: `web: gunicorn video_processor_api:app --bind 0.0.0.0:$PORT --workers 2 --timeout 300`

2. **requirements.txt** ✓
   - Lists all Python dependencies
   - Railway will automatically install these

3. **Main Python file** ✓
   - `video_processor_api.py` - Your Flask app

## 🚀 Railway Auto-Detection

Railway will automatically:
- Detect Python project
- Install dependencies from requirements.txt
- Use Procfile to start the app
- Assign a PORT environment variable
- Generate a public URL

## ⚙️ Important Notes

### Port Configuration
- Railway automatically sets `PORT` environment variable
- Your app uses: `port = int(os.environ.get('PORT', 9001))`
- This is correct! ✓

### Workers
- Your Procfile uses `--workers 2`
- This is good for Railway's free tier
- Can increase for paid plans

### Timeout
- Set to 300 seconds (5 minutes)
- Good for video processing
- Railway allows long-running requests

## 📦 Large Files Warning

Your project has some large files:
- `yolov8s.pt` (22MB)
- `nitymed_resnet18.pth` (44MB)
- Various video files

**Important:** 
- Railway has a 500MB slug size limit
- Consider using `.gitignore` to exclude:
  - Test videos (*.mp4, *.avi, etc.)
  - Temporary files
  - Large datasets

## 🔧 Recommended .gitignore Additions

Add these to your `.gitignore`:

```
# Large files
*.mp4
*.avi
*.mov
*.mkv
*.m4v
*.webm
*.flv
*.wmv

# Temporary directories
/tmp/
/uploads/
/detected_frames_*/
/microsleep_frames/
/phone_detection_output/
/output/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
myenv/
mp_env/

# OS
.DS_Store
Thumbs.db
```

## 🎯 Deployment Steps

1. **Clean up large files** (optional but recommended):
   ```bash
   # Remove test videos and outputs before pushing
   rm -rf detected_frames_* microsleep_frames output.mp4
   ```

2. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Ready for Railway deployment"
   git push origin main
   ```

3. **Deploy on Railway**:
   - Railway will automatically build and deploy
   - Watch the build logs for any errors
   - First deployment takes 2-5 minutes

## 🐛 Common Issues

### Issue: Build fails due to size
**Solution:** Add large files to .gitignore and push again

### Issue: App crashes on start
**Solution:** Check Railway logs for Python errors

### Issue: Dependencies fail to install
**Solution:** Verify requirements.txt has correct package names

### Issue: Timeout errors
**Solution:** Already handled with `--timeout 300` in Procfile

## 📊 Monitoring

After deployment:
- Check Railway dashboard for logs
- Monitor memory usage (free tier: 512MB)
- Watch for crashes or restarts

## 💰 Railway Pricing

- **Free Tier**: $5 credit/month (good for testing)
- **Hobby Plan**: $5/month (better for production)
- Your app should work fine on free tier for testing

## 🎉 Success Indicators

Your deployment is successful when:
- ✅ Build completes without errors
- ✅ App shows "Running" status
- ✅ Health check endpoint responds: `https://your-app.railway.app/api/health`
- ✅ You can access: `https://your-app.railway.app/`

## 🔗 Useful Railway Commands

```bash
# Install Railway CLI (optional)
npm i -g @railway/cli

# Login
railway login

# Link to project
railway link

# View logs
railway logs

# Open in browser
railway open
```
