#!/usr/bin/env python3
"""
Music Graph RAG Demo
Shows comparison between Naive RAG vs Graph RAG with LLM reranking
Supports both local music files and demo datasets
"""

from music_graph_rag import MusicGraphRAG
from akriti_real_music_data import AKRITI_REAL_MUSIC_DATA

# Demo dataset based on your Spotify interface
SPOTIFY_DEMO_DATA = [
    {
        "passage": "Post Malone is a prominent artist in your music collection, appearing both in popular radio stations and top artists. He spans multiple genres including Hip-Hop, Pop, and Rock, collaborating with artists like Morgan Wallen and Juice WRLD. His versatile style bridges mainstream pop and hip-hop, making him a central figure in contemporary music.",
        "triplets": [
            ["Post Malone", "appears in", "popular radio stations"],
            ["Post Malone", "is in", "top artists"],
            ["Post Malone", "spans genres", "Hip-Hop"],
            ["Post Malone", "spans genres", "Pop"],
            ["Post Malone", "spans genres", "Rock"],
            ["Post Malone", "collaborates with", "Morgan Wallen"],
            ["Post Malone", "collaborates with", "Juice WRLD"],
            ["Post Malone", "represents", "contemporary music"],
        ],
    },
    {
        "passage": "Drake is a major influence in your hip-hop listening habits, featured prominently in both radio stations and top artists. His radio station includes collaborations with PARTYNEXTDOOR, Kendrick Lamar, and Brent Favre, showing his connection to the broader hip-hop and R&B ecosystem. Drake represents the mainstream hip-hop sound that forms a significant part of your musical taste.",
        "triplets": [
            ["Drake", "is featured in", "radio stations"],
            ["Drake", "is in", "top artists"],
            ["Drake", "collaborates with", "PARTYNEXTDOOR"],
            ["Drake", "collaborates with", "Kendrick Lamar"],
            ["Drake", "collaborates with", "Brent Favre"],
            ["Drake", "represents", "mainstream hip-hop"],
            ["Drake", "influences", "hip-hop listening habits"],
        ],
    },
    {
        "passage": "Taylor Swift represents your pop music preferences, appearing in both radio and top artists sections. Her presence indicates an appreciation for narrative songwriting and pop craftsmanship. She connects to artists like Harry Styles, Chappell Roan, and Billie Eilish in the broader pop ecosystem you enjoy.",
        "triplets": [
            ["Taylor Swift", "appears in", "radio stations"],
            ["Taylor Swift", "is in", "top artists"],
            ["Taylor Swift", "represents", "pop music preferences"],
            ["Taylor Swift", "is known for", "narrative songwriting"],
            ["Taylor Swift", "connects to", "Harry Styles"],
            ["Taylor Swift", "connects to", "Chappell Roan"],
            ["Taylor Swift", "connects to", "Billie Eilish"],
            ["Taylor Swift", "represents", "pop craftsmanship"],
        ],
    },
    {
        "passage": "Your music collection reveals a deep appreciation for Sufi and South Asian music, with artists like Kaavish representing Pakistani Sufi rock, Qurat-ul-Ain Balouch bringing contemporary Sufi vocals, and the legendary Abida Parveen representing classical Sufi traditions. This shows your connection to spiritual and culturally rich musical traditions from the Indian subcontinent.",
        "triplets": [
            ["You", "listen to", "Sufi music"],
            ["You", "appreciate", "South Asian music"],
            ["Kaavish", "represents", "Pakistani Sufi rock"],
            ["Qurat-ul-Ain Balouch", "performs", "contemporary Sufi"],
            ["Abida Parveen", "represents", "classical Sufi tradition"],
            ["Your taste", "includes", "spiritual music"],
            ["Your taste", "connects to", "Indian subcontinent culture"],
        ],
    },
    {
        "passage": "Kaavish appears prominently in your music collection with multiple tracks including 'Faasle', 'Tere Pyar Main', 'O Yaara', and 'Moray Saiyaan'. This Pakistani Sufi rock band combines traditional Sufi poetry with modern rock arrangements, representing your taste for music that bridges classical and contemporary styles. Your engagement with Kaavish shows appreciation for Urdu poetry and Sufi philosophy set to accessible musical arrangements.",
        "triplets": [
            ["Kaavish", "is prominent in", "your music collection"],
            ["Kaavish", "performs", "Faasle"],
            ["Kaavish", "performs", "Tere Pyar Main"],
            ["Kaavish", "performs", "O Yaara"],
            ["Kaavish", "performs", "Moray Saiyaan"],
            ["Kaavish", "combines", "Sufi poetry with rock"],
            ["Kaavish", "represents", "classical-contemporary bridge"],
            ["You", "appreciate", "Urdu poetry"],
            ["You", "appreciate", "Sufi philosophy"],
        ],
    },
    {
        "passage": "Your genre diversity spans Hip-Hop (Drake, Kendrick Lamar, Post Malone), Pop (Taylor Swift, Billie Eilish, Lady Gaga), R&B (SZA, The Weeknd), Classic Rock (Fleetwood Mac), Country (Morgan Wallen), and Sufi music (Kaavish, Abida Parveen). This shows a broad musical palate that appreciates both mainstream and alternative approaches within each genre, as well as cross-cultural musical traditions.",
        "triplets": [
            ["Your taste", "includes", "Hip-Hop"],
            ["Your taste", "includes", "Pop"],
            ["Your taste", "includes", "R&B"],
            ["Your taste", "includes", "Classic Rock"],
            ["Your taste", "includes", "Country"],
            ["Your taste", "includes", "Sufi music"],
            ["Your taste", "appreciates", "mainstream music"],
            ["Your taste", "appreciates", "alternative approaches"],
            ["Your taste", "shows", "broad musical palate"],
            ["Your taste", "includes", "cross-cultural traditions"],
        ],
    },
]

def main():
    print("🎵 Music Graph RAG Demo - Akriti's REAL Music Taste")
    print("=" * 60)
    print("📊 Compare Naive RAG vs Graph RAG with LLM Reranking")
    print("Based on your ACTUAL Spotify playlists:")
    print("🏠 House: RÜFÜS DU SOL, Lane 8, Swedish House Mafia, Elderbrook")
    print("🎸 Indie: Arctic Monkeys, Radiohead, Bon Iver, Grizzly Bear, Death Cab")
    print("🤠 Country: Morgan Wallen, Brooks & Dunn, Dan + Shay, Old Dominion")
    print("=" * 60)
    
    # Initialize with YOUR REAL music data
    music_rag = MusicGraphRAG(demo_data=AKRITI_REAL_MUSIC_DATA, collection_prefix="akriti_real")
    
    try:
        # Setup the system
        music_rag.build_graph_structure(verbose=True)
        music_rag.create_embeddings_and_store()
        
        print("\n🚀 System ready!")
        
        # Interactive query mode with comparison
        print("\n" + "="*60)
        print("🎤 Interactive Music Query with Method Comparison")
        print("💡 Example queries:")
        examples = [
            "What genres of music do I listen to?",
            "Tell me about my Sufi music preferences",
            "How does Drake connect to my other hip-hop artists?",
            "What does my music taste say about cultural diversity?",
            "Which artists span multiple genres in my collection?"
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example}")
        
        print("\n🎯 Commands:")
        print("   • Type your question for Graph RAG vs Naive RAG comparison")
        print("   • Type 'graph [query]' for Graph RAG only")
        print("   • Type 'naive [query]' for Naive RAG only")
        print("   • Type 'quit' to exit")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n🎵 Your music question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for exploring Graph RAG!")
                    break
                
                if not user_input:
                    continue
                
                # Parse command and query
                if user_input.lower().startswith('graph '):
                    query = user_input[6:]
                    print(f"\n🕸️ Graph RAG Only:")
                    answer = music_rag.query_music_knowledge_with_llm_reranking(query, compare_with_naive=False)
                    print(f"{answer}")
                    
                elif user_input.lower().startswith('naive '):
                    query = user_input[6:]
                    print(f"\n📊 Naive RAG Only:")
                    answer = music_rag._naive_rag_retrieval(query)
                    context = "\n\n".join(answer)
                    final_answer = music_rag._generate_final_answer(query, context, "Naive RAG")
                    print(f"{final_answer}")
                    
                else:
                    # Default: show comparison
                    print(f"\n⚖️ Comparing Graph RAG vs Naive RAG:")
                    answer = music_rag.query_music_knowledge_with_llm_reranking(user_input, compare_with_naive=True)
                    print(f"{answer}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        print("💡 Make sure your OpenAI API key is set correctly in config.py")

def run_local_music_demo():
    """Example of how to use with local music files"""
    print("🎵 Local Music Files Demo")
    print("=" * 50)
    
    # Replace with your actual music directories
    music_directories = [
        "/Users/yourname/Music",
        "/Users/yourname/iTunes/iTunes Media/Music",
    ]
    
    music_rag = MusicGraphRAG(music_directories=music_directories, collection_prefix="local")
    music_rag.build_graph_structure()
    music_rag.create_embeddings_and_store()
    
    # Example query
    result = music_rag.query_music_knowledge_with_llm_reranking(
        "What genres of music do I have in my collection?", 
        compare_with_naive=True
    )
    print(result)

if __name__ == "__main__":
    main()
    
    # Uncomment to test with local music files
    # run_local_music_demo()
