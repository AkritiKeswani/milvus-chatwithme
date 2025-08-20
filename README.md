# Milvus Chat with Me - Music Graph RAG

A Next.js web application that showcases Graph RAG (Retrieval-Augmented Generation) technology powered by Milvus vector database. Chat with Akriti about her music taste using advanced AI knowledge retrieval!

## 🚀 **Quick Start (Local Development)**

### Prerequisites
- Node.js 18+ 
- Python 3.8+
- OpenAI API key

### Setup
1. **Clone and install dependencies:**
   ```bash
   git clone <your-repo>
   cd milvus-chatwithme
   
   # Install Python dependencies
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
   # Install Node.js dependencies
   cd web
   npm install
   ```

2. **Set up environment variables:**
   ```bash
   # In the root directory, create .env file
   echo "OPENAI_API_KEY=your_actual_openai_api_key_here" > .env
   ```

3. **Run the application:**
   ```bash
   # Terminal 1: Start Python backend (Graph RAG)
   cd ..  # Back to root
   source venv/bin/activate
   python web_query.py
   
   # Terminal 2: Start Next.js frontend
   cd web
   npm run dev
   ```

4. **Open your browser:** http://localhost:3000

## 🌐 **Deploy to Vercel (Production)**

### Option 1: Vercel Dashboard (Recommended)

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Ready for Vercel deployment"
   git push origin main
   ```

2. **Deploy on Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Sign in with GitHub
   - Click "New Project"
   - Import your repository
   - Set the **Root Directory** to `web`
   - Click "Deploy"

3. **Add Environment Variables:**
   - In your Vercel project dashboard
   - Go to **Settings** → **Environment Variables**
   - Add: `OPENAI_API_KEY` = `your_actual_openai_api_key_here`
   - Redeploy

### Option 2: Vercel CLI

1. **Install Vercel CLI:**
   ```bash
   npm i -g vercel
   ```

2. **Deploy:**
   ```bash
   cd web
   vercel
   # Follow prompts, set root directory to current folder
   # Add environment variables when prompted
   ```

3. **Set environment variables:**
   ```bash
   vercel env add OPENAI_API_KEY
   # Enter your OpenAI API key
   vercel --prod  # Deploy to production
   ```

## 🔧 **How It Works**

### Architecture
- **Frontend:** Next.js 15 with TypeScript
- **Backend:** Python Graph RAG system via API routes
- **AI Engine:** OpenAI GPT models for intelligent responses
- **Knowledge Base:** Milvus vector database for music knowledge

### Graph RAG Process
1. **Entity Extraction:** Identifies artists/genres in user queries
2. **Subgraph Expansion:** Traverses relationships in knowledge graph
3. **Context Retrieval:** Gets relevant passages from Milvus
4. **Response Generation:** Creates intelligent answers using AI

## 📁 **Project Structure**

```
milvus-chatwithme/
├── web/                    # Next.js frontend (deploy this to Vercel)
│   ├── src/app/api/chat/  # Chat API endpoint
│   ├── src/app/page.tsx   # Main chat interface
│   └── vercel.json        # Vercel configuration
├── music_graph_rag.py     # Core Graph RAG implementation
├── web_query.py           # Python backend for web queries
├── config.py              # OpenAI and Milvus configuration
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (local only)
```

## 🌍 **Environment Variables**

### Local Development (.env file)
```bash
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### Vercel Deployment
- Set `OPENAI_API_KEY` in Vercel dashboard
- **Never commit your .env file to Git!**

## 🎯 **Features**

- **Intelligent Music Chat:** Ask about artists, genres, and music trends
- **Graph RAG Technology:** Advanced knowledge retrieval using Milvus
- **Real-time Responses:** Instant AI-powered answers
- **Modern UI:** Clean, responsive chat interface
- **Vercel Ready:** Fully deployable serverless architecture

## 🚨 **Important Notes**

- **Cost Efficiency:** Uses OpenAI API efficiently with smart query handling
- **Security:** API keys are stored securely in environment variables
- **Scalability:** Built for Vercel's serverless architecture
- **Performance:** Optimized for fast response times

## 🆘 **Troubleshooting**

### Common Issues
1. **"next: command not found"** → Run `npm install` in web directory
2. **Python import errors** → Activate virtual environment first
3. **OpenAI API errors** → Check API key and billing status
4. **Vercel deployment fails** → Ensure root directory is set to `web`

### Support
- Check Vercel deployment logs for errors
- Verify environment variables are set correctly
- Ensure all dependencies are installed

---

**Ready to deploy?** Follow the Vercel deployment steps above and your Graph RAG music chat will be live on the internet! 🎵✨
