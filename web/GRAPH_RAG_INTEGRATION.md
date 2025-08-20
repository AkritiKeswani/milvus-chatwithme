# Integrating Your Graph RAG System with Vercel

## Current Status
✅ **Web app is working** with intelligent music responses  
✅ **Ready for Vercel deployment**  
🔄 **Graph RAG integration pending** (needs OpenAI quota)  

## What's Working Now
The web app currently uses intelligent pattern matching to provide personalized music responses. It's much better than the simple if-else statements and gives responses that feel like they come from someone who really knows music.

## How to Get Your Full Graph RAG System Working

### Option 1: Fix OpenAI Quota (Recommended)
1. **Check your OpenAI account**: Go to [platform.openai.com](https://platform.openai.com)
2. **Add credits**: Purchase credits or upgrade your plan
3. **Update the route**: Replace the current logic with your Graph RAG system

### Option 2: Use Alternative AI Provider
If OpenAI is too expensive, you can integrate with:
- **Anthropic Claude** (often cheaper)
- **Google Gemini** (free tier available)
- **Local models** (for development)

## Integration Steps

### Step 1: Fix OpenAI Quota
```bash
# Check your current OpenAI status
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.openai.com/v1/models
```

### Step 2: Update the Chat Route
Once OpenAI is working, replace the current logic in `web/src/app/api/chat/route.ts` with:

```typescript
// Import your Graph RAG system
import { MusicGraphRAG } from '../../../music_graph_rag';
import { AKRITI_REAL_MUSIC_DATA } from '../../../akriti_real_music_data';

// Initialize RAG system
let rag: MusicGraphRAG | null = null;

async function initializeRAG() {
  if (rag) return rag;
  
  try {
    rag = new MusicGraphRAG(demo_data=AKRITI_REAL_MUSIC_DATA, collection_prefix='akriti_vercel');
    await rag.build_graph_structure(verbose=False);
    await rag.create_embeddings_and_store();
    return rag;
  } catch (error) {
    console.error('RAG initialization failed:', error);
    return null;
  }
}

// In your POST handler:
const ragSystem = await initializeRAG();
if (ragSystem) {
  const result = await ragSystem.query_music_knowledge_with_llm_reranking(message, compare_with_naive=True);
  return NextResponse.json({ response: result });
}
```

### Step 3: Copy Required Files
```bash
# Copy your Graph RAG files to the web directory
cp music_graph_rag.py web/src/app/api/chat/
cp akriti_real_music_data.py web/src/app/api/chat/
cp config.py web/src/app/api/chat/
```

### Step 4: Update Vercel Config
Add Python runtime to `vercel.json`:
```json
{
  "functions": {
    "src/app/api/chat/route.ts": {
      "runtime": "python3.9"
    }
  }
}
```

## Current Architecture
```
┌─────────────────┐
│   Next.js App  │ ← Deployed on Vercel
│   (Port 3000)  │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Chat API      │ ← Intelligent responses (ready for Graph RAG)
│  (route.ts)    │
└─────────────────┘
```

## Deployment Status
- ✅ **Frontend**: Ready for Vercel
- ✅ **Chat API**: Working with intelligent responses
- 🔄 **Graph RAG**: Ready to integrate once OpenAI quota is fixed
- ✅ **Database**: Will use Vercel's serverless environment

## Testing Your Current Setup
```bash
# Test the chat API
curl -X POST "http://localhost:3000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What kind of music do you like?"}'

# Test different genres
curl -X POST "http://localhost:3000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about electronic music"}'
```

## Next Steps
1. **Deploy to Vercel** (current version works great!)
2. **Fix OpenAI quota** 
3. **Integrate full Graph RAG system**
4. **Enjoy your AI-powered music chat!**

Your app is already much better than the simple version and ready for production use!
