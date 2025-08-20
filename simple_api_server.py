#!/usr/bin/env python3
"""
Simple API server for testing - no OpenAI embeddings required
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Simple Music Chat API", version="1.0.0")

# Add CORS middleware for web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Simple Music Chat API is running"}

@app.post("/query")
async def query_music(request: QueryRequest):
    try:
        message = request.message.lower()
        
        # Simple response logic without OpenAI
        if "rock" in message or "indie" in message:
            response = "I love indie rock! Some of my favorite artists include Radiohead, The National, and Fleet Foxes. I'm really into the atmospheric, introspective sound that defines the genre."
        elif "folk" in message or "acoustic" in message:
            response = "Folk music has a special place in my heart. I adore artists like Sufjan Stevens, Bon Iver, and Fleet Foxes. There's something about acoustic guitars and storytelling that really speaks to me."
        elif "electronic" in message or "ambient" in message:
            response = "I'm fascinated by electronic and ambient music! Tame Impala, Beach House, and The xx create these incredible soundscapes that I can get lost in for hours."
        elif "jazz" in message or "blues" in message:
            response = "Jazz and blues are incredible! I love how they capture raw emotion and improvisation. Miles Davis, John Coltrane, and Nina Simone are some of my favorites."
        elif "pop" in message:
            response = "Pop music can be really sophisticated! I appreciate artists like Taylor Swift who blend catchy melodies with thoughtful lyrics. It's not just about the hooks - there's real artistry there."
        elif "classical" in message:
            response = "Classical music is timeless! I love how it can be both complex and emotionally direct. Beethoven, Debussy, and Bach are incredible for different moods."
        else:
            response = "I love talking about music! I have pretty eclectic taste - from indie rock to folk to electronic. What kind of music are you into? I'm always looking for new recommendations!"
        
        return {"response": response}
        
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "api": "simple_music_chat",
        "openai_required": False
    }

if __name__ == "__main__":
    print("Starting Simple Music Chat API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
