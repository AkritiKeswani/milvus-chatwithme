# Vercel Deployment Guide

## Overview
This is a Next.js web app that simulates chatting with Akriti about music taste using OpenAI's GPT-4.

## Prerequisites
- Vercel account
- OpenAI API key

## Deployment Steps

### 1. Set up Environment Variables in Vercel
1. Go to your Vercel project dashboard
2. Navigate to Settings → Environment Variables
3. Add: `OPENAI_API_KEY` with your actual OpenAI API key

### 2. Deploy to Vercel
```bash
# Install Vercel CLI (if not already installed)
npm i -g vercel

# Deploy from the web directory
cd web
vercel

# Follow the prompts to connect to your Vercel account
```

### 3. Alternative: Deploy via GitHub
1. Push your code to GitHub
2. Connect your GitHub repo to Vercel
3. Vercel will automatically deploy on push

## How It Works
- **Frontend**: Next.js React app with chat interface
- **Backend**: Vercel serverless functions using OpenAI API
- **No Python backend needed**: Everything runs on Vercel's infrastructure

## Local Development
```bash
cd web
npm install
npm run dev
```

The app will run on http://localhost:3000

## Notes
- The Python RAG backend is not used in this deployment
- All AI responses come directly from OpenAI GPT-4
- The system prompt simulates Akriti's music knowledge and personality
