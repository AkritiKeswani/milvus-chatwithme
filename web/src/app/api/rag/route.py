#!/usr/bin/env python3
"""
Python API route for Vercel that runs your Graph RAG system
This allows Vercel to run your Graph RAG backend directly
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to Python path so we can import your modules
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from music_graph_rag import MusicGraphRAG
    from akriti_real_music_data import AKRITI_REAL_MUSIC_DATA
    GRAPH_RAG_AVAILABLE = True
except ImportError as e:
    print(f"Graph RAG not available: {e}")
    GRAPH_RAG_AVAILABLE = False

# Global RAG system (initialized once per serverless function instance)
rag = None
rag_initialized = False

def initialize_rag():
    """Initialize the RAG system when first needed"""
    global rag, rag_initialized
    
    if not GRAPH_RAG_AVAILABLE:
        return None
        
    if rag_initialized and rag:
        return rag
    
    try:
        print("Initializing Graph RAG system...")
        rag = MusicGraphRAG(demo_data=AKRITI_REAL_MUSIC_DATA, collection_prefix='akriti_vercel')
        rag.build_graph_structure(verbose=False)
        rag.create_embeddings_and_store()
        rag_initialized = True
        print("Graph RAG system ready!")
        return rag
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        rag_initialized = False
        return None

def query_music(message):
    """Query the music knowledge using Graph RAG"""
    try:
        # Initialize RAG system if needed
        rag_system = initialize_rag()
        
        if not rag_system:
            return {
                "response": "I'm having trouble accessing my advanced music knowledge right now. This might be due to OpenAI API limits or system initialization issues.",
                "error": "RAG system unavailable",
                "fallback": True
            }
        
        # Use your existing Graph RAG system
        result = rag_system.query_music_knowledge_with_llm_reranking(
            message, 
            compare_with_naive=True
        )
        
        # Clean up the result
        if result.startswith('[Graph RAG]'):
            result = result.replace('[Graph RAG]', '').strip()
        
        return {"response": result, "source": "Graph RAG"}
        
    except Exception as e:
        print(f"Graph RAG query error: {e}")
        # Fallback response if RAG fails
        fallback_response = "I'm having trouble processing that specific question with my advanced analysis, but I love talking about music! I'm really into indie rock, folk, and alternative music. What genres are you curious about?"
        return {"response": fallback_response, "error": str(e), "fallback": True}

# Vercel serverless function entry point
def handler(request):
    """Main handler for Vercel serverless function"""
    try:
        # Parse request
        if request.method == 'POST':
            body = request.get_json()
            message = body.get('message', '') if body else ''
            
            if not message:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'Message is required'})
                }
            
            # Query the music knowledge
            result = query_music(message)
            
            return {
                'statusCode': 200,
                'body': json.dumps(result),
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            }
        
        elif request.method == 'GET':
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Graph RAG API is running. Send a POST request with a message to use the advanced music analysis system.',
                    'status': 'Graph RAG available' if GRAPH_RAG_AVAILABLE else 'Graph RAG not available'
                }),
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
            }
        
        else:
            return {
                'statusCode': 405,
                'body': json.dumps({'error': 'Method not allowed'})
            }
            
    except Exception as e:
        print(f"API Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'response': "Sorry, I'm having some technical difficulties right now. Try asking me about my music taste in a bit!",
                'error': str(e)
            }),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }

# For local testing
if __name__ == "__main__":
    # Simulate a request for testing
    class MockRequest:
        def __init__(self, method, body=None):
            self.method = method
            self.body = body
        
        def get_json(self):
            return json.loads(self.body) if self.body else {}
    
    # Test the handler
    test_request = MockRequest('POST', '{"message": "What kind of music do you like?"}')
    result = handler(test_request)
    print(json.dumps(result, indent=2))
