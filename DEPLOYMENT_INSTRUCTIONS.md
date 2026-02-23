# 🚀 Deployment Instructions

## ✅ Prerequisites
- GitHub account
- Railway account (https://railway.app)
- Vercel account (https://vercel.com)

---

## 📦 Part 1: Deploy Backend to Railway

### Step 1: Push Backend to GitHub

```bash
cd "/Users/yakesh/Desktop/Railways Project"

# Initialize git if not already done
git init
git add .
git commit -m "Backend ready for deployment"

# Create a new repository on GitHub, then:
git remote add origin YOUR_GITHUB_REPO_URL
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Railway

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your backend repository
5. Railway will auto-detect Python and use your `Procfile`
6. Wait for deployment to complete
7. **Copy your Railway URL** (e.g., `https://your-app.railway.app`)

### Step 3: Configure Railway (Optional)

- Go to your project → Variables
- Railway automatically sets `PORT` - no need to add it
- Your app will run on the port Railway assigns

---

## 🎨 Part 2: Deploy Frontend to Vercel

### Step 1: Update Environment Variable

1. Open `.env.production` in `pilot-eye-analytics-hub` folder
2. Replace `https://your-backend-url.railway.app` with your actual Railway URL
3. Save the file

### Step 2: Push Frontend to GitHub

```bash
cd "/Users/yakesh/Desktop/Railways Project/pilot-eye-analytics-hub"

# Initialize git if not already done
git init
git add .
git commit -m "Frontend ready for deployment"

# Create a new repository on GitHub, then:
git remote add origin YOUR_FRONTEND_GITHUB_REPO_URL
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Vercel

1. Go to https://vercel.com
2. Click "Add New..." → "Project"
3. Import your frontend repository
4. Configure:
   - Framework Preset: **Vite**
   - Build Command: `npm run build`
   - Output Directory: `dist`
   - Install Command: `npm install`
5. Add Environment Variable:
   - Key: `VITE_API_URL`
   - Value: Your Railway backend URL (e.g., `https://your-app.railway.app`)
6. Click "Deploy"
7. Wait for deployment to complete
8. **Copy your Vercel URL** (e.g., `https://your-app.vercel.app`)

---

## 🔗 Part 3: Update Backend CORS (Important!)

After deploying frontend, update your backend to allow requests from Vercel:

1. Open `video_processor_api.py`
2. Find the line: `CORS(app)`
3. Update it to:
```python
CORS(app, origins=["https://your-app.vercel.app", "http://localhost:5173"])
```
4. Commit and push changes to GitHub
5. Railway will automatically redeploy

---

## ✅ Testing Your Deployment

1. Open your Vercel URL in browser
2. Try uploading a video
3. Check if it processes correctly
4. Monitor Railway logs for any errors

---

## 🐛 Troubleshooting

### Backend Issues:
- Check Railway logs: Project → Deployments → View Logs
- Ensure all files are committed to GitHub
- Verify `requirements.txt` has all dependencies
- Check if `Procfile` is correct

### Frontend Issues:
- Check Vercel deployment logs
- Verify `VITE_API_URL` environment variable is set
- Check browser console for CORS errors
- Ensure Railway backend is running

### CORS Errors:
- Update backend CORS settings to include your Vercel URL
- Redeploy backend after CORS changes

---

## 📝 Important Files

### Backend (Railway):
- `video_processor_api.py` - Main Flask app
- `Procfile` - Tells Railway how to run the app
- `requirements.txt` - Python dependencies

### Frontend (Vercel):
- `package.json` - Node dependencies and build scripts
- `vercel.json` - Vercel configuration
- `.env.production` - Production environment variables
- `src/services/api.ts` - API client (uses VITE_API_URL)

---

## 🎉 Success!

Once both are deployed:
- Frontend: `https://your-app.vercel.app`
- Backend: `https://your-app.railway.app`

Your Railway project is now live and accessible worldwide! 🚀
