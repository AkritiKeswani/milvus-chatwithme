# Music Graph RAG System 🎵

A Graph-based Retrieval Augmented Generation system for exploring and querying your personal music collection. This system adapts the graph RAG approach to create a knowledge graph from your music directories, allowing you to ask natural language questions about your musical interests and collection.

## Features

- 🎵 **Automatic Music Discovery**: Scans your music directories and extracts metadata from audio files
- 🕸️ **Knowledge Graph Construction**: Creates relationships between artists, albums, genres, and songs
- 🧠 **Semantic Search**: Uses embeddings to understand and answer natural language queries
- 💾 **Persistent Storage**: Stores knowledge graph in Milvus vector database
- 🎯 **Interactive Queries**: Ask questions like "What genres do I listen to most?" or "Which artists span multiple genres?"

## How It Works

The system follows the same graph RAG pattern as the Bernoulli family example but applies it to music:

1. **Data Extraction**: Scans your music directories and extracts metadata from audio files
2. **Triplet Creation**: Creates subject-predicate-object relationships like:
   - `["The Beatles", "created album", "Abbey Road"]`
   - `["Abbey Road", "is of genre", "Rock"]`
   - `["The Beatles", "performs in genre", "Rock"]`
3. **Graph Construction**: Builds entity and relation mappings similar to the original example
4. **Embedding Generation**: Creates vector embeddings for all entities, relations, and passages
5. **Query Processing**: Answers natural language questions using semantic search and LLM reasoning

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

### Quick Start
```bash
python run_music_rag.py
```

This will:
1. Scan your music directories
2. Build the knowledge graph
3. Create and store embeddings in Milvus
4. Start an interactive query session

### Example Queries

- "What genres of music do I listen to most?"
- "Which artists have multiple albums in my collection?"
- "Tell me about my jazz music collection"
- "What albums do I have by The Beatles?"
- "Show me artists that span multiple genres"
- "What's the diversity of my music taste?"

### Programmatic Usage

```python
from music_graph_rag import MusicGraphRAG

# Initialize with your music directories
music_rag = MusicGraphRAG(["/path/to/your/music"])

# Build the graph
music_rag.build_graph_structure()

# Create embeddings and store
music_rag.create_embeddings_and_store()

# Query your collection
answer = music_rag.query_music_knowledge("What genres do I listen to?")
print(answer)
```

## File Structure

- `config.py` - Configuration for OpenAI API and Milvus client
- `music_graph_rag.py` - Main Music Graph RAG implementation
- `run_music_rag.py` - Simple script to run the system
- `requirements.txt` - Python dependencies
- `music_graph_data.json` - Generated graph data (after first run)
- `milvus.db` - Local Milvus database file

## Supported Audio Formats

The system supports common audio formats including:
- MP3 (.mp3)
- FLAC (.flac)
- WAV (.wav)
- M4A (.m4a)
- AAC (.aac)
- OGG (.ogg)
- WMA (.wma)

## Graph Structure

The system creates the same type of entity-relation mappings as the Bernoulli example:

- **Entities**: Artists, albums, songs, genres, directory names
- **Relations**: Relationships like "performs", "created album", "belongs to genre", etc.
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
