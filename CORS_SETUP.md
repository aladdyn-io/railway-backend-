# CORS Configuration for Production

## Current Setup
Your backend currently has: `CORS(app)` which allows all origins.

## After Deploying Frontend to Vercel

Once you have your Vercel URL, update `video_processor_api.py`:

### Find this line (around line 29):
```python
CORS(app)
```

### Replace with:
```python
# Allow requests from Vercel frontend and localhost for development
CORS(app, origins=[
    "https://your-app.vercel.app",  # Replace with your actual Vercel URL
    "http://localhost:5173",         # For local development
    "http://localhost:3000"          # Alternative local port
])
```

### Or keep it simple for now:
```python
# Allow all origins (simpler but less secure)
CORS(app, origins="*")
```

## Steps:
1. Deploy backend to Railway first
2. Deploy frontend to Vercel
3. Get your Vercel URL
4. Update CORS in video_processor_api.py with your Vercel URL
5. Commit and push - Railway will auto-redeploy

## Note:
For development/testing, you can keep `CORS(app)` as is.
For production, it's better to specify allowed origins for security.
