#!/usr/bin/env python3
import sys
import os
import json
from io import StringIO

def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python web_query.py 'your query'"}))
        return
    
    query = sys.argv[1]
    
    try:
        from akriti_real_music_data import AKRITI_REAL_MUSIC_DATA
        from music_graph_rag import MusicGraphRAG
        
        # Initialize the system (suppress output)
        
        # Capture stdout to hide setup messages
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            rag = MusicGraphRAG(demo_data=AKRITI_REAL_MUSIC_DATA, collection_prefix='akriti_web')
            rag.build_graph_structure(verbose=False)
            rag.create_embeddings_and_store()
        finally:
            sys.stdout = old_stdout
        
        # Capture ALL output including debug info - just like CLI!
        captured_output = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            # Run the EXACT same method as CLI with comparison
            result = rag.query_music_knowledge_with_llm_reranking(
                query, 
                compare_with_naive=True  # This gives us the full CLI experience!
            )
        finally:
            sys.stdout = old_stdout
        
        # Get all the beautiful debug output
        debug_output = captured_output.getvalue()
        
        # Clean up the final result
        if result.startswith('[Graph RAG]'):
            result = result.replace('[Graph RAG]', '').strip()
        
        # Combine debug info + final result - EXACTLY like CLI
        full_response = debug_output + "\n" + result if debug_output.strip() else result
        
        print(json.dumps({"response": full_response}))
        
    except Exception as e:
        error_msg = f"Sorry, I had trouble processing that question about my music taste. Could you try rephrasing it? (Error: {str(e)})"
        print(json.dumps({"response": error_msg}))

if __name__ == "__main__":
    main()
