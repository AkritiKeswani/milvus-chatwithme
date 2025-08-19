# Music Graph RAG Web Interface 🎵

This is a [Next.js](https://nextjs.org) web interface for the Music Graph RAG System, allowing users to chat about music taste through an intelligent conversational interface.

## Features

- 🎯 **Interactive Chat Interface**: Ask questions about music preferences
- 🕸️ **Graph RAG Backend**: Powered by the complete Graph RAG pipeline
- ⚖️ **Method Comparison**: Shows Graph RAG vs Naive RAG results
- 🎵 **Music Intelligence**: Understands artists, genres, and musical relationships
- 🚀 **Real-time Processing**: Instant responses through API integration

## Local Development Setup

### Prerequisites
- Node.js 18+
- Python 3.8+ with virtual environment set up (see main README)
- OpenAI API key

### 1. Install Dependencies
```bash
npm install
```

### 2. Environment Configuration

Create a `.env.local` file in the web directory:
```bash
echo "OPENAI_API_KEY=your-actual-api-key-here" > .env.local
```

**Important**: The web app requires the Python backend to be properly configured with a virtual environment and the same OpenAI API key.

### 3. Start Development Server

**CRITICAL**: You must source the parent environment file to ensure the Python backend has access to environment variables:

```bash
# From the web directory
source ../.env && npm run dev
```

This ensures:
- Next.js loads environment variables from `.env.local`
- Python backend (called via API) has access to the same variables
- Both frontend and backend can communicate properly

### 4. Access the Application

Open [http://localhost:3000](http://localhost:3000) with your browser to start chatting about music preferences.

## Architecture

- **Frontend**: Next.js React app with chat interface
- **API**: `/api/chat` endpoint that bridges to Python backend  
- **Backend**: Python `web_query.py` script using the MusicGraphRAG system
- **Data**: Uses real music data with Graph RAG processing

## Troubleshooting

### Common Issues

1. **API Key Errors**: 
   - Ensure `.env.local` exists with valid `OPENAI_API_KEY`
   - Make sure you source `../.env` when starting the dev server

2. **Python Backend Errors**:
   - Verify the virtual environment is set up in the parent directory
   - Check that `web_query.py` works independently: `python web_query.py "test"`

3. **Milvus Connection Issues**:
   - The `milvus.db` file should exist in the parent directory
   - Run the demo script first to initialize the database

### Environment Setup Verification

Test that everything is working:
```bash
# Test Python backend directly
cd .. && source venv/bin/activate && python web_query.py "what music do you like?"

# Test web API
curl -X POST http://localhost:3000/api/chat -H "Content-Type: application/json" -d '{"message": "test"}'
```

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
