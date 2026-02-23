# 🚀 Step-by-Step Deployment Guide
## Frontend on Vercel + Backend on Railway

---

## Part 1: Deploy Backend to Railway

### Step 1: Prepare GitHub Repository (Backend)

1. Go to **https://github.com** and sign in
2. Create a new repository (e.g. `cvvrs-backend`) or use existing
3. On your Mac, open Terminal and run:

```bash
cd "/Users/yakesh/Desktop/Railways Project"

# If you don't have git initialized:
git init

# Add files (backend files only - don't include pilot-eye-analytics-hub in backend repo)
# OR if your whole project is one repo, that's fine too

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/cvvrs-backend.git

# Add, commit, push
git add .
git commit -m "Backend ready for Railway deployment"
git branch -M main
git push -u origin main
```

**Backend repo must include:**
- `video_processor_api.py`
- `requirements.txt`
- `Procfile`
- `railway.json`
- `perfect_hand_signal_detector.py`
- `station_alert_system.py`
- `requirement_team_loader.py`
- Excel files: `Detected_Signals_Lat_Long.xlsx`, `Detected_Signals_Lat_Long_Enhanced.xlsx`
- Any other Python files the API needs

---

### Step 2: Deploy Backend on Railway

1. Go to **https://railway.app**
2. Click **Login** → Sign in with **GitHub**
3. Click **New Project**
4. Select **Deploy from GitHub repo**
5. Choose your backend repository (e.g. `cvvrs-backend`)
6. If asked for **Root Directory**:
   - If backend is at repo root → leave blank
   - If backend is in a subfolder (e.g. `backend/`) → enter that path
7. Railway will auto-detect Python and use `Procfile`
8. Click **Deploy** — wait 5–10 minutes (first run downloads YOLO model ~22MB)

---

### Step 3: Get Railway API URL

1. In Railway project, click your **service**
2. Go to **Settings** tab
3. Under **Networking**, click **Generate Domain**
4. Copy the URL (e.g. `https://cvvrs-backend-production-xxxx.up.railway.app`)
5. **Save this URL** — you’ll need it for the frontend

---

### Step 4: Configure Railway (Optional)

- **Environment variables**: Usually none needed; `PORT` is set automatically
- **Timeout**: Video processing can take a few minutes; 300 seconds is usually enough

---

## Part 2: Deploy Frontend to Vercel

### Step 5: Prepare GitHub Repository (Frontend)

**Option A – Frontend in its own repo**

1. Create a repo for the frontend (e.g. `cvvrs-frontend`)
2. Push only the `pilot-eye-analytics-hub` folder:

```bash
cd "/Users/yakesh/Desktop/Railways Project/pilot-eye-analytics-hub"
git init
git remote add origin https://github.com/YOUR_USERNAME/cvvrs-frontend.git
git add .
git commit -m "Frontend ready for Vercel"
git branch -M main
git push -u origin main
```

**Option B – Frontend in same repo as backend**

1. Use the same repo as backend
2. Vercel will use a **Root Directory** setting (see Step 7)

---

### Step 6: Deploy Frontend on Vercel

1. Go to **https://vercel.com**
2. Click **Login** → Sign in with **GitHub**
3. Click **Add New...** → **Project**
4. Import your repository
5. Select the repo that contains the frontend (or the monorepo)

---

### Step 7: Configure Vercel Project

| Setting | Value |
|--------|--------|
| **Framework Preset** | Vite |
| **Root Directory** | `pilot-eye-analytics-hub` (if frontend is in subfolder, otherwise leave blank) |
| **Build Command** | `npm run build` |
| **Output Directory** | `dist` |
| **Install Command** | `npm install` |

---

### Step 8: Add Environment Variable (IMPORTANT)

1. Before deploying, expand **Environment Variables**
2. Add:
   - **Name**: `VITE_API_URL`
   - **Value**: `https://YOUR-RAILWAY-URL.up.railway.app` (from Step 3)
   - **Environments**: Production, Preview, Development (select all)
3. Click **Deploy**

---

### Step 9: Wait for Build

- Vercel will run `npm install` and `npm run build`
- Usually takes 1–2 minutes
- Your frontend will be live at something like `https://cvvrs-frontend.vercel.app`

---

## Part 3: Verify Everything Works

### Step 10: Test the Setup

1. Open your Vercel URL in a browser
2. Upload a video
3. Processing should hit your Railway backend
4. Check that results appear (phone, bags, hand signals, etc.)

### Troubleshooting

| Issue | What to do |
|-------|------------|
| **"Failed to fetch" / CORS** | Backend already has CORS enabled. Confirm `VITE_API_URL` in Vercel matches Railway URL exactly |
| **Backend timeout** | In Railway, check logs; long videos may need a longer timeout |
| **Frontend shows old API** | Redeploy Vercel after changing `VITE_API_URL` (env vars require a new build) |
| **404 on refresh** | `vercel.json` rewrites should fix this; ensure it’s in the frontend root |

---

## Quick Checklist

- [ ] Backend pushed to GitHub
- [ ] Backend deployed on Railway
- [ ] Railway domain generated and URL copied
- [ ] Frontend pushed to GitHub (or same repo)
- [ ] Frontend deployed on Vercel
- [ ] `VITE_API_URL` set in Vercel
- [ ] Video upload and processing tested

---

## Summary of URLs

| Service | Where | Example |
|---------|-------|---------|
| **Backend (Railway)** | Settings → Networking → Domain | `https://xxx.up.railway.app` |
| **Frontend (Vercel)** | Project → Domains | `https://your-app.vercel.app` |
| **Env var** | Vercel → Settings → Environment Variables | `VITE_API_URL` = Railway URL |
