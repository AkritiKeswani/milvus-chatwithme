#!/usr/bin/env python3
"""
Quick demo of your musical taste based on visible Spotify interface
This will work until you get your official Spotify data export
"""

from spotify_demo_data import SpotifyDemoRAG

def main():
    print("🎵 Spotify Musical Taste Demo")
    print("=" * 50)
    print("Based on your Spotify interface showing:")
    print("📻 Radio: SZA, Drake, Fleetwood Mac, Sabrina Carpenter, Post Malone, Taylor Swift, Morgan Wallen")
    print("⭐ Top Artists: Post Malone, The Weeknd, Kendrick Lamar, Billie Eilish, Lady Gaga, Drake, Taylor Swift")
    print("=" * 50)
    
    # Initialize the demo system
    spotify_rag = SpotifyDemoRAG()
    
    try:
        # Build graph and embeddings
        print("\n🔨 Building knowledge graph from Spotify data...")
        spotify_rag.build_graph_from_spotify_data()
        
        print("\n🧠 Creating embeddings...")
        spotify_rag.create_embeddings_and_store()
        
        print("\n✅ Demo setup complete!")
        
        # Interactive query mode
        print("\n" + "="*50)
        print("🎤 Ask about your musical taste!")
        print("Examples:")
        print("  • What genres do I like?")
        print("  • Tell me about my hip-hop taste")
        print("  • How diverse is my music?")
        print("  • What artists connect my different genres?")
        print("  • How does Drake relate to my other artists?")
        print("\nCommands:")
        print("  • Type 'advanced [query]' for subgraph expansion")
        print("  • Type 'simple [query]' for basic similarity search")
        print("  • Type 'quit' to exit")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n🎵 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for exploring your musical taste!")
                    break
                
                if not user_input:
                    continue
                
                # Parse command and query
                if user_input.lower().startswith('advanced '):
                    query = user_input[9:]  # Remove 'advanced ' prefix
                    print(f"\n🕸️ Using advanced subgraph expansion...")
                    answer = spotify_rag.query_spotify_knowledge_advanced(query)
                    print(f"\n🎯 Advanced Insight: {answer}")
                    
                elif user_input.lower().startswith('simple '):
                    query = user_input[7:]  # Remove 'simple ' prefix
                    print(f"\n📊 Using simple similarity search...")
                    answer = spotify_rag.query_spotify_knowledge(query)
                    print(f"\n🎯 Simple Insight: {answer}")
                    
                else:
                    # Default to advanced method
                    print(f"\n🔍 Analyzing with advanced graph methods...")
                    answer = spotify_rag.query_spotify_knowledge_advanced(user_input)
                    print(f"\n🎯 Insight: {answer}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        print("💡 Make sure your OpenAI API key is set in config.py")

if __name__ == "__main__":
    main()
