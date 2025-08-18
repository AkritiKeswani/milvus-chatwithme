# Music Graph RAG System 🎵

A complete Graph-based Retrieval Augmented Generation system for exploring your musical taste and collection. Features **side-by-side comparison** of Naive RAG vs Graph RAG with LLM reranking, showing the power of knowledge graphs for music discovery.

## Features

- 🕸️ **Complete Graph RAG Pipeline**: NER → Subgraph Expansion → LLM Reranking
- ⚖️ **Method Comparison**: See Graph RAG vs Naive RAG results side-by-side
- 🎵 **Music Intelligence**: Understands artists, genres, cultural connections, and musical relationships
- 💾 **Milvus Integration**: Efficient vector storage with 3 separate collections (entities, relations, passages)
- 🧠 **LLM Reranking**: Uses Chain-of-Thought reasoning to select most relevant relationships
- 🎯 **Interactive Demo**: Ready-to-run with Spotify-based demo data

## How Graph RAG Works

Unlike traditional RAG that just searches text chunks, Graph RAG builds a knowledge graph:

1. **Knowledge Graph**: Creates entities (artists, genres) and relations (collaborations, influences)
2. **NER Extraction**: Finds relevant entities in your query
3. **Subgraph Expansion**: Uses adjacency matrices to discover connected relationships  
4. **Milvus Search**: Vector similarity search within the expanded subgraph
5. **LLM Reranking**: Selects most relevant relations using reasoning
6. **Final Answer**: Generates response from carefully selected context

**Example**: Query *"What Sufi music do I like?"* → Finds Kaavish entity → Expands to Sufi relationships → Ranks by relevance → Returns rich cultural context

## Technical Implementation

This implementation follows the complete [Milvus Graph RAG tutorial](https://milvus.io/docs/graph_rag_with_milvus.md) with all advanced features:

- **Adjacency Matrices**: Proper entity-relation, entity-entity, and relation-relation matrices using `scipy.sparse.csr_matrix`
- **Multi-degree Expansion**: Matrix multiplication for 1-degree, 2-degree, or higher graph traversal
- **Similarity Thresholds**: Configurable filtering with `entity_sim_thresh` and `relation_sim_thresh`
- **Correct ID Mapping**: Proper mapping between candidate list indices and original relation IDs in LLM reranking
- **Three-Collection Architecture**: Separate Milvus collections for optimal graph construction and retrieval

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure OpenAI API Key**:
   Edit `config.py` and replace the placeholder with your actual OpenAI API key:
   ```python
   os.environ["OPENAI_API_KEY"] = "your-actual-api-key-here"
   ```

3. **Set Your Music Directories**:
   Edit `run_music_rag.py` and update the `music_directories` list with your actual music folder paths:
   ```python
   music_directories = [
       "/Users/yourname/Music",
       "/Users/yourname/iTunes/iTunes Media/Music",
       "/Volumes/External/Music",
       # Add your music directories here
   ]
   ```

## Usage

### Quick Demo
```bash
python run_demo.py
```

This will:
1. Load demo dataset with your musical preferences  
2. Build the knowledge graph (entities, relations, passages)
3. Create embeddings and store in Milvus (3 collections)
4. Start interactive comparison mode (Graph RAG vs Naive RAG)

### Example Queries

- "What genres of music do I listen to most?"
- "Which artists have multiple albums in my collection?"
- "What genres does Post Malone span?"
- "Tell me about my Pakistani artists"
- "Show me artists that span multiple genres"
- "What's the diversity of my music taste?"

### Programmatic Usage

```python
from music_graph_rag import MusicGraphRAG
from run_demo import SPOTIFY_DEMO_DATA

# Initialize with demo data (currently the only supported input)
music_rag = MusicGraphRAG(demo_data=SPOTIFY_DEMO_DATA)

# Build the graph and create embeddings
music_rag.build_graph_structure()
music_rag.create_embeddings_and_store()

# Query with Graph RAG vs Naive RAG comparison
answer = music_rag.query_music_knowledge_with_llm_reranking(
    "What genres do I listen to?", 
    compare_with_naive=True
)
print(answer)
```

## File Structure

- `config.py` - Configuration for OpenAI API and Milvus client
- `music_graph_rag.py` - Main Music Graph RAG implementation
- `run_demo.py` - Interactive demo script
- `requirements.txt` - Python dependencies
- `milvus.db` - Local Milvus database file (created on first run)

## Current Data Source

**Note**: This implementation currently works with structured demo data only (Spotify-based musical preferences). It does **not** scan local music files or directories. The demo data includes:

- Artists: Post Malone, Drake, Taylor Swift, Kaavish, Abida Parveen, etc.
- Genres: Hip-Hop, Pop, Sufi, R&B, Classic Rock, Country
- Cultural connections and musical relationships
- **Passages**: Descriptive text about each musical relationship
- **Mappings**: 
  - `entityid_2_relationids`: Maps entity IDs to related relation IDs
  - `relationid_2_passageids`: Maps relation IDs to passage IDs

## Customization

You can extend the system by:

1. **Adding new relationship types** in `_create_passage_from_metadata()`
2. **Modifying metadata extraction** in `_extract_audio_metadata()`
3. **Customizing the query prompts** in `query_music_knowledge()`
4. **Adding new audio formats** in `_is_audio_file()`

## Troubleshooting

- **No music found**: Make sure your directory paths are correct and contain audio files
- **Metadata extraction fails**: Some files may not have proper ID3 tags or metadata
- **OpenAI API errors**: Check that your API key is valid and has sufficient credits
- **Milvus connection issues**: The system uses local Milvus Lite by default, which should work out of the box

## Examples from Your Collection

Once set up, you can ask questions like:

> **Query**: "What genres of music do I listen to most?"
> 
> **Response**: "Based on your music collection, you primarily listen to Rock (45% of collection), Pop (25%), Jazz (15%), Classical (10%), and Electronic (5%). Your rock collection includes artists like The Beatles, Led Zeppelin, and Pink Floyd, with albums spanning from the 1960s to modern rock."

> **Query**: "Tell me about artists that span multiple genres"
> 
> **Response**: "Several artists in your collection work across multiple genres: David Bowie appears in both Rock and Electronic categories, Miles Davis spans Jazz and Fusion, and Radiohead bridges Alternative Rock and Electronic music. This shows your appreciation for artists who experiment with different musical styles."
