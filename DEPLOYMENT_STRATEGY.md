# Deployment Strategy for Graph RAG Music Chat

## Overview
This project has **two deployment options** that both preserve your Graph RAG backend:

1. **Hybrid Deployment (Recommended)**: Vercel frontend + Python API server
2. **Local Development**: Full local setup with Graph RAG

## Option 1: Hybrid Deployment (Vercel + Python API)

### Architecture
```
┌─────────────────┐    HTTP    ┌──────────────────┐
│   Vercel App   │ ──────────→ │  Python API     │
│   (Next.js)    │             │  (Graph RAG)    │
└─────────────────┘             └──────────────────┘
```

### Steps:

#### 1. Deploy Python API Server
You have several options for the Python backend:

**Option A: Railway (Recommended)**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy from project root
railway login
railway init
railway up
```

**Option B: Render**
- Create account on render.com
- Connect your GitHub repo
- Set build command: `pip install -r requirements.txt`
- Set start command: `python api_server.py`

**Option C: Heroku**
```bash
# Create Procfile
echo "web: python api_server.py" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

#### 2. Deploy Vercel Frontend
```bash
cd web
vercel
```

#### 3. Set Environment Variables
In Vercel dashboard:
- `GRAPH_RAG_API_URL`: Your deployed Python API URL
- `OPENAI_API_KEY`: Your OpenAI API key

## Option 2: Local Development (Current Setup)

### Run Both Services Locally
```bash
# Terminal 1: Start Python API server
source venv/bin/activate
python api_server.py

# Terminal 2: Start Next.js app
cd web
npm run dev
```

## Testing Your Graph RAG

### Test the API Server
```bash
# Start the API server
source venv/bin/activate
python api_server.py

# Test with curl
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"message": "What kind of music do you like?"}'
```

### Test the Web App
1. Start the API server (Terminal 1)
2. Start the web app (Terminal 2)
3. Open http://localhost:3000
4. Chat with your Graph RAG system!

## What This Gives You

✅ **Your Graph RAG backend is fully preserved**
✅ **All your music knowledge and embeddings**
✅ **Vercel deployment capability**
✅ **Local development still works**
✅ **No need to manually activate virtual environment**

## File Structure
```
milvus-chatwithme/
├── .env                          ← Environment variables
├── api_server.py                 ← NEW: FastAPI server for Graph RAG
├── music_graph_rag.py            ← Your existing Graph RAG system
├── akriti_real_music_data.py    ← Your music data
├── web/                          ← Next.js frontend
│   ├── src/app/api/chat/        ← Updated to call your API
│   └── vercel.json              ← Vercel config
└── venv/                         ← Python environment
```

## Next Steps

1. **Test locally first**: Run both services to ensure everything works
2. **Choose deployment platform**: Railway, Render, or Heroku for Python API
3. **Deploy Python API**: Get your API URL
4. **Deploy to Vercel**: Set the API URL in environment variables
5. **Enjoy your Graph RAG-powered music chat!**

Your Graph RAG system is now accessible via HTTP API and can be deployed anywhere Python runs!
