# 🚀 Quick Deployment Checklist

## ✅ Backend (Railway) - 5 Steps

1. [ ] Push code to GitHub
   ```bash
   cd "/Users/yakesh/Desktop/Railways Project"
   git init
   git add .
   git commit -m "Backend deployment"
   git remote add origin YOUR_REPO_URL
   git push -u origin main
   ```

2. [ ] Go to https://railway.app → New Project → Deploy from GitHub

3. [ ] Select your backend repository

4. [ ] Wait for deployment (Railway auto-detects Python + Procfile)

5. [ ] Copy Railway URL: `https://your-app.railway.app` ✍️

---

## ✅ Frontend (Vercel) - 6 Steps

1. [ ] Update `.env.production` with Railway URL
   ```
   VITE_API_URL=https://your-app.railway.app
   ```

2. [ ] Push code to GitHub
   ```bash
   cd "/Users/yakesh/Desktop/Railways Project/pilot-eye-analytics-hub"
   git init
   git add .
   git commit -m "Frontend deployment"
   git remote add origin YOUR_FRONTEND_REPO_URL
   git push -u origin main
   ```

3. [ ] Go to https://vercel.com → Add New → Project

4. [ ] Import your frontend repository

5. [ ] Add Environment Variable:
   - Key: `VITE_API_URL`
   - Value: Your Railway URL

6. [ ] Click Deploy and wait

---

## ✅ Final Test

1. [ ] Open Vercel URL in browser
2. [ ] Upload a test video
3. [ ] Verify it processes correctly
4. [ ] Check Railway logs if issues occur

---

## 📝 URLs to Save

- Backend (Railway): `_______________________`
- Frontend (Vercel): `_______________________`

---

## 🆘 Need Help?

- Railway logs: Project → Deployments → View Logs
- Vercel logs: Project → Deployments → View Function Logs
- Check browser console (F12) for frontend errors
- Check CORS_SETUP.md if you get CORS errors

---

## 🎉 Done!

Your app is now live and accessible worldwide!
