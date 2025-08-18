#!/usr/bin/env python3
"""
Simple script to run the Music Graph RAG system
Replace the music_directories with your actual music folder paths
"""

from music_graph_rag import MusicGraphRAG

def main():
    # REPLACE THESE PATHS WITH YOUR ACTUAL MUSIC DIRECTORIES
    music_directories = [
        "/Users/akritiwork/Music",  # Default macOS Music folder
        # Add more paths like:
        # "/Users/akritiwork/Desktop/Music",
        # "/Volumes/External/Music",
        # "/Users/akritiwork/Documents/iTunes/iTunes Media/Music",
    ]
    
    print("🎵 Starting Music Graph RAG System...")
    print(f"📁 Scanning directories: {music_directories}")
    
    # Initialize the system
    music_rag = MusicGraphRAG(music_directories)
    
    try:
        # Build the graph structure
        print("\n🔨 Building graph structure...")
        music_rag.build_graph_structure()
        
        # Create embeddings and store in Milvus
        print("\n🧠 Creating embeddings and storing in Milvus...")
        music_rag.create_embeddings_and_store()
        
        # Save graph data for later analysis
        print("\n💾 Saving graph data...")
        music_rag.save_graph_data()
        
        print("\n✅ Setup complete! You can now query your music collection.")
        print("\n🎯 Example queries you can try:")
        example_queries = [
            "What genres of music do I listen to most?",
            "Which artists have multiple albums in my collection?",
            "Tell me about my rock music collection",
            "What's the most common genre in my library?",
            "Show me artists that span multiple genres"
        ]
        
        for i, query in enumerate(example_queries, 1):
            print(f"  {i}. {query}")
        
        # Interactive query loop
        print("\n" + "="*50)
        print("🎤 Interactive Music Query Mode")
        print("Type 'quit' or 'exit' to stop")
        print("="*50)
        
        while True:
            try:
                user_query = input("\n🎵 Ask about your music: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if not user_query:
                    continue
                    
                print(f"\n🔍 Searching for: {user_query}")
                answer = music_rag.query_music_knowledge(user_query)
                print(f"\n🎯 Answer: {answer}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("💡 Make sure your music directories exist and contain audio files")
        print("💡 Check that your OpenAI API key is set correctly in config.py")

if __name__ == "__main__":
    main()
