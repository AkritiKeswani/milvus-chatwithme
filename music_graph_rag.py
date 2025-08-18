"""
Complete Graph RAG System for Music Collections
Supports both local music files and Spotify data with full Milvus integration
"""

import os
import json
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Any, Set, Optional
from scipy.sparse import csr_matrix
import mutagen
from mutagen.id3 import ID3NoHeaderError
from config import milvus_client, llm, embedding_model
from tqdm import tqdm
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

class MusicGraphRAG:
    def __init__(self, music_directories: Optional[List[str]] = None, demo_data: Optional[List[Dict]] = None, collection_prefix: str = "music"):
        """
        Initialize the Music Graph RAG system
        
        Args:
            music_directories: List of paths to your music directories (for local files)
            demo_data: Pre-built dataset (for Spotify demo or custom data)
            collection_prefix: Prefix for Milvus collections
        """
        self.music_directories = music_directories or []
        self.demo_data = demo_data
        self.collection_prefix = collection_prefix
        self.music_dataset = []
        self.entities = []
        self.relations = []
        self.passages = []
        self.entityid_2_relationids = defaultdict(list)
        self.relationid_2_passageids = defaultdict(list)
        
        # Milvus collection names
        self.entity_col_name = f"{collection_prefix}_entities"
        self.relation_col_name = f"{collection_prefix}_relations"  
        self.passage_col_name = f"{collection_prefix}_passages"
        
        # Adjacency matrices for subgraph expansion
        self.entity_relation_adj = None
        self.entity_adj_1_degree = None
        self.relation_adj_1_degree = None
        
    def scan_music_directories(self) -> List[Dict[str, Any]]:
        """
        Scan music directories and extract metadata to create passages and triplets
        """
        music_data = []
        
        for music_dir in self.music_directories:
            music_dir_path = Path(music_dir)
            if not music_dir_path.exists():
                print(f"Directory {music_dir} does not exist, skipping...")
                continue
                
            print(f"Scanning directory: {music_dir}")
            
            # Walk through the directory structure
            for root, dirs, files in os.walk(music_dir_path):
                root_path = Path(root)
                
                # Extract information from directory structure
                relative_path = root_path.relative_to(music_dir_path)
                path_parts = list(relative_path.parts)
                
                # Look for audio files in this directory
                audio_files = [f for f in files if self._is_audio_file(f)]
                
                if audio_files:
                    # Extract metadata from audio files
                    for audio_file in audio_files[:5]:  # Limit to first 5 files per directory
                        file_path = root_path / audio_file
                        metadata = self._extract_audio_metadata(file_path)
                        
                        if metadata:
                            passage_info = self._create_passage_from_metadata(
                                metadata, path_parts, str(file_path)
                            )
                            if passage_info:
                                music_data.append(passage_info)
                                
                    # Also create directory-level information
                    if len(path_parts) >= 2:  # Artist/Album structure
                        dir_passage = self._create_directory_passage(path_parts, audio_files)
                        if dir_passage:
                            music_data.append(dir_passage)
        
        return music_data
    
    def _is_audio_file(self, filename: str) -> bool:
        """Check if file is an audio file"""
        audio_extensions = {'.mp3', '.flac', '.wav', '.m4a', '.aac', '.ogg', '.wma'}
        return Path(filename).suffix.lower() in audio_extensions
    
    def _extract_audio_metadata(self, file_path: Path) -> Dict[str, str]:
        """Extract metadata from audio file"""
        try:
            audiofile = mutagen.File(file_path)
            if audiofile is None:
                return {}
                
            metadata = {}
            
            # Common tags across formats
            tag_mapping = {
                'artist': ['TPE1', 'ARTIST', '\xa9ART', 'Artist'],
                'album': ['TALB', 'ALBUM', '\xa9alb', 'Album'],
                'title': ['TIT2', 'TITLE', '\xa9nam', 'Title'],
                'genre': ['TCON', 'GENRE', '\xa9gen', 'Genre'],
                'date': ['TDRC', 'DATE', '\xa9day', 'Date'],
                'albumartist': ['TPE2', 'ALBUMARTIST', 'aART', 'AlbumArtist']
            }
            
            for field, possible_keys in tag_mapping.items():
                for key in possible_keys:
                    if key in audiofile:
                        value = audiofile[key]
                        if isinstance(value, list) and value:
                            metadata[field] = str(value[0])
                        else:
                            metadata[field] = str(value)
                        break
            
            return metadata
            
        except (ID3NoHeaderError, Exception) as e:
            return {}
    
    def _create_passage_from_metadata(self, metadata: Dict[str, str], 
                                    path_parts: List[str], file_path: str) -> Dict[str, Any]:
        """Create a passage and triplets from audio metadata"""
        if not metadata.get('artist') or not metadata.get('title'):
            return None
            
        artist = metadata.get('artist', 'Unknown Artist')
        title = metadata.get('title', 'Unknown Title')
        album = metadata.get('album', path_parts[-1] if path_parts else 'Unknown Album')
        genre = metadata.get('genre', 'Unknown Genre')
        
        # Create passage description
        passage = f"{artist} performs '{title}' from the album '{album}'. "
        if genre != 'Unknown Genre':
            passage += f"This song belongs to the {genre} genre. "
        if len(path_parts) >= 2:
            passage += f"The file is organized under {'/'.join(path_parts)}. "
        passage += f"File location: {file_path}"
        
        # Create triplets
        triplets = [
            [artist, "performs", title],
            [title, "is from album", album],
            [artist, "created album", album],
        ]
        
        if genre != 'Unknown Genre':
            triplets.extend([
                [title, "belongs to genre", genre],
                [artist, "performs in genre", genre],
                [album, "is of genre", genre]
            ])
            
        # Add directory structure relationships
        if len(path_parts) >= 1:
            triplets.append([artist, "is organized under", path_parts[0]])
        if len(path_parts) >= 2:
            triplets.append([album, "is in directory", path_parts[1]])
            
        return {
            "passage": passage,
            "triplets": triplets
        }
    
    def _create_directory_passage(self, path_parts: List[str], audio_files: List[str]) -> Dict[str, Any]:
        """Create passage from directory structure"""
        if len(path_parts) < 2:
            return None
            
        artist_dir = path_parts[0]
        album_dir = path_parts[1]
        
        passage = f"The artist directory '{artist_dir}' contains the album '{album_dir}' "
        passage += f"with {len(audio_files)} tracks. "
        
        if len(path_parts) > 2:
            passage += f"It's further organized in subdirectory: {'/'.join(path_parts[2:])}. "
            
        triplets = [
            [artist_dir, "has album", album_dir],
            [album_dir, "contains", f"{len(audio_files)} tracks"],
            [artist_dir, "is organized in", "music collection"]
        ]
        
        return {
            "passage": passage,
            "triplets": triplets
        }
    
    def build_graph_structure(self, verbose: bool = True):
        """Build the graph structure from music data or demo data"""
        if verbose:
            print("🔨 Building knowledge graph...")
        
        # Use demo data if provided, otherwise scan directories
        if self.demo_data:
            self.music_dataset = self.demo_data
            if verbose:
                print("   Using provided demo dataset")
        else:
            self.music_dataset = self.scan_music_directories()
            if verbose:
                print(f"   Scanned {len(self.music_directories)} directories")
        
        # Build entities, relations, and mappings
        for passage_id, dataset_info in enumerate(self.music_dataset):
            passage, triplets = dataset_info["passage"], dataset_info["triplets"]
            self.passages.append(passage)
            
            for triplet in triplets:
                # Add entities (subjects and objects)
                if triplet[0] not in self.entities:
                    self.entities.append(triplet[0])
                if triplet[2] not in self.entities:
                    self.entities.append(triplet[2])
                
                # Create relation string
                relation = " ".join(triplet)
                if relation not in self.relations:
                    self.relations.append(relation)
                    
                    # Map entity IDs to relation IDs
                    subject_id = self.entities.index(triplet[0])
                    object_id = self.entities.index(triplet[2])
                    relation_id = len(self.relations) - 1
                    
                    self.entityid_2_relationids[subject_id].append(relation_id)
                    self.entityid_2_relationids[object_id].append(relation_id)
                
                # Map relation ID to passage ID
                relation_id = self.relations.index(relation)
                if passage_id not in self.relationid_2_passageids[relation_id]:
                    self.relationid_2_passageids[relation_id].append(passage_id)
        
        if verbose:
            print(f"   ✅ Built graph with {len(self.entities)} entities, {len(self.relations)} relations, {len(self.passages)} passages")
    
    def create_embeddings_and_store(self):
        """Create embeddings and store in Milvus following the proper pattern"""
        print("Creating embeddings and storing in Milvus...")
        
        # Get embedding dimension
        embedding_dim = len(embedding_model.embed_query("foo"))
        
        # Create separate collections for entities, relations, and passages
        self._create_milvus_collections(embedding_dim)
        
        # Insert data using the proper batch insertion method
        self._milvus_insert(
            collection_name=self.entity_col_name,
            text_list=self.entities,
        )
        
        self._milvus_insert(
            collection_name=self.relation_col_name,
            text_list=self.relations,
        )
        
        self._milvus_insert(
            collection_name=self.passage_col_name,
            text_list=self.passages,
        )
        
        print("Successfully stored embeddings in separate Milvus collections")
        
        # Build adjacency matrices for subgraph expansion
        self._build_adjacency_matrices()
    
    def _create_milvus_collections(self, embedding_dim: int):
        """Create separate Milvus collections for entities, relations, and passages"""
        self.entity_col_name = "music_entity_collection"
        self.relation_col_name = "music_relation_collection"  
        self.passage_col_name = "music_passage_collection"
        
        def create_milvus_collection(collection_name: str):
            if milvus_client.has_collection(collection_name=collection_name):
                milvus_client.drop_collection(collection_name=collection_name)
            milvus_client.create_collection(
                collection_name=collection_name,
                dimension=embedding_dim,
                consistency_level="Bounded",
            )
        
        create_milvus_collection(self.entity_col_name)
        create_milvus_collection(self.relation_col_name)
        create_milvus_collection(self.passage_col_name)
        
        print(f"Created collections: {self.entity_col_name}, {self.relation_col_name}, {self.passage_col_name}")
    
    def _milvus_insert(self, collection_name: str, text_list: List[str]):
        """Insert data into Milvus collection with proper batching"""
        batch_size = 512
        for row_id in tqdm(range(0, len(text_list), batch_size), desc=f"Inserting into {collection_name}"):
            batch_texts = text_list[row_id : row_id + batch_size]
            batch_embeddings = embedding_model.embed_documents(batch_texts)

            batch_ids = [row_id + j for j in range(len(batch_texts))]
            batch_data = [
                {
                    "id": id_,
                    "text": text,
                    "vector": vector,
                }
                for id_, text, vector in zip(batch_ids, batch_texts, batch_embeddings)
            ]
            milvus_client.insert(
                collection_name=collection_name,
                data=batch_data,
            )
    
    def _build_adjacency_matrices(self):
        """Build adjacency matrices for efficient subgraph expansion"""
        print("Building adjacency matrices for subgraph expansion...")
        
        # Construct the adjacency matrix of entities and relations
        self.entity_relation_adj = np.zeros((len(self.entities), len(self.relations)))
        for entity_id, entity in enumerate(self.entities):
            if entity_id in self.entityid_2_relationids:
                self.entity_relation_adj[entity_id, self.entityid_2_relationids[entity_id]] = 1

        # Convert to sparse matrix for efficient computation
        self.entity_relation_adj = csr_matrix(self.entity_relation_adj)

        # Construct 1-degree entity-entity and relation-relation adjacency matrices
        self.entity_adj_1_degree = self.entity_relation_adj @ self.entity_relation_adj.T
        self.relation_adj_1_degree = self.entity_relation_adj.T @ self.entity_relation_adj
        
        print("Adjacency matrices built successfully")
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Simple NER extraction for music entities from query
        In practice, you could use a more sophisticated NER model
        """
        # Simple approach: look for known entities in the query
        query_lower = query.lower()
        query_entities = []
        
        # Check for entities in the query
        for entity in self.entities:
            if entity.lower() in query_lower:
                query_entities.append(entity)
        
        # If no entities found, use keyword extraction
        if not query_entities:
            # Look for music-related keywords that might match entities
            music_keywords = ['artist', 'song', 'album', 'genre', 'band', 'music', 'track']
            words = query_lower.split()
            
            for word in words:
                # Find entities that contain this word
                for entity in self.entities:
                    if word in entity.lower() and entity not in query_entities:
                        query_entities.append(entity)
                        break
        
        return query_entities[:3]  # Limit to top 3 for efficiency
    
    def _expand_subgraph(self, query_entities: List[str], query: str, 
                        target_degree: int = 1, top_k: int = 3,
                        entity_sim_filter_thresh: float = 0.0,
                        relation_sim_filter_thresh: float = 0.0) -> List[str]:
        """
        Expand subgraph using adjacency matrices to get candidate relations
        """
        if self.entity_relation_adj is None:
            self._build_adjacency_matrices()
        
        # Get embeddings for query entities
        query_ner_embeddings = [
            embedding_model.embed_query(query_ner) for query_ner in query_entities
        ]
        
        # Search for similar entities
        entity_search_res = []
        if query_ner_embeddings:
            entity_search_res = milvus_client.search(
                collection_name=self.entity_col_name,
                data=query_ner_embeddings,
                limit=top_k,
                output_fields=["id"],
            )
        
        # Search for similar relations
        query_embedding = embedding_model.embed_query(query)
        relation_search_res = milvus_client.search(
            collection_name=self.relation_col_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["id"],
        )[0]
        
        # Compute target degree adjacency matrices
        entity_adj_target_degree = self.entity_adj_1_degree
        for _ in range(target_degree - 1):
            entity_adj_target_degree = entity_adj_target_degree * self.entity_adj_1_degree
            
        relation_adj_target_degree = self.relation_adj_1_degree
        for _ in range(target_degree - 1):
            relation_adj_target_degree = relation_adj_target_degree * self.relation_adj_1_degree

        # CRITICAL: Compute entity_relation_adj_target_degree properly  
        entity_relation_adj_target_degree = entity_adj_target_degree @ self.entity_relation_adj
        
        # Expand relations from relation search results
        expanded_relations_from_relation = set()
        
        filtered_hit_relation_ids = [
            relation_res["entity"]["id"] for relation_res in relation_search_res
            if relation_res['distance'] > relation_sim_filter_thresh
        ]
        
        for hit_relation_id in filtered_hit_relation_ids:
            if hit_relation_id < relation_adj_target_degree.shape[0]:
                expanded_relations_from_relation.update(
                    relation_adj_target_degree[hit_relation_id].nonzero()[1].tolist()
                )
        
        # Expand relations from entity search results  
        expanded_relations_from_entity = set()
        
        filtered_hit_entity_ids = [
            one_entity_res["entity"]["id"]
            for one_entity_search_res in entity_search_res
            for one_entity_res in one_entity_search_res
            if one_entity_res['distance'] > entity_sim_filter_thresh
        ]
        
        for filtered_hit_entity_id in filtered_hit_entity_ids:
            if filtered_hit_entity_id < entity_relation_adj_target_degree.shape[0]:
                expanded_relations_from_entity.update(
                    entity_relation_adj_target_degree[filtered_hit_entity_id].nonzero()[1].tolist()
                )
        
        # Merge the expanded relations from the relation and entity retrieval ways
        relation_candidate_ids = list(
            expanded_relations_from_relation | expanded_relations_from_entity
        )
        
        # Get relation texts (filter valid IDs)
        valid_candidate_pairs = [
            (relation_id, self.relations[relation_id]) 
            for relation_id in relation_candidate_ids
            if relation_id < len(self.relations)
        ]
        
        if valid_candidate_pairs:
            relation_candidate_ids, relation_candidate_texts = zip(*valid_candidate_pairs)
            return list(relation_candidate_texts), list(relation_candidate_ids)
        else:
            return [], []
    
    def _rerank_relations_with_llm(self, query: str, relation_candidate_texts: List[str], 
                                 relation_candidate_ids: List[int], top_k: int = 3) -> List[int]:
        """
        Use LLM to rerank and select the most relevant relations using Chain-of-Thought reasoning
        """
        if not relation_candidate_texts:
            return []
        
        # Create relation description string with CANDIDATE LIST indices (not global IDs)
        relation_des_str = "\n".join(
            f"[{idx}] {text}" for idx, text in enumerate(relation_candidate_texts)
        ).strip()
        
        # One-shot example for music domain
        query_prompt_one_shot_input = """I will provide you with a list of musical relationship descriptions. Your task is to select up to 3 relationships that may be useful to answer the given question about music preferences and listening habits. Please return a JSON object containing your thought process and a list of the selected relationships in order of their relevance.

Question:
What genres does Post Malone span and how does he connect to other artists?

Relationship descriptions:
[1] Post Malone spans genres Hip-Hop
[2] Post Malone spans genres Pop
[3] Post Malone spans genres Rock
[4] Post Malone collaborates with Morgan Wallen
[5] Post Malone collaborates with Juice WRLD
[6] Post Malone represents contemporary music
[7] Drake represents mainstream hip-hop
[8] Taylor Swift represents pop music preferences
"""
        
        query_prompt_one_shot_output = """{"thought_process": "To answer the question about Post Malone's genres and connections, I need to identify relationships that show: 1) what genres he spans, and 2) his collaborations with other artists. The most relevant relationships are those that directly mention Post Malone's genre diversity and his collaborations.", "useful_relationships": ["[1] Post Malone spans genres Hip-Hop", "[2] Post Malone spans genres Pop", "[4] Post Malone collaborates with Morgan Wallen"]}"""
        
        query_prompt_template = """Question:
{question}

Relationship descriptions:
{relation_des_str}

"""
        
        # Create the reranking prompt chain
        rerank_prompts = ChatPromptTemplate.from_messages([
            HumanMessage(query_prompt_one_shot_input),
            AIMessage(query_prompt_one_shot_output),
            HumanMessagePromptTemplate.from_template(query_prompt_template),
        ])
        
        rerank_chain = (
            rerank_prompts
            | llm.bind(response_format={"type": "json_object"})
            | JsonOutputParser()
        )
        
        try:
            rerank_res = rerank_chain.invoke({
                "question": query,
                "relation_des_str": relation_des_str
            })
            
            rerank_relation_ids = []
            rerank_relation_lines = rerank_res.get("useful_relationships", [])
            
            for line in rerank_relation_lines[:top_k]:  # Limit to top_k
                try:
                    # Extract candidate list index from format "[idx] text"
                    start_idx = line.find("[") + 1
                    end_idx = line.find("]")
                    if start_idx > 0 and end_idx > start_idx:
                        candidate_idx = int(line[start_idx:end_idx])
                        # Map candidate index back to original relation ID
                        if 0 <= candidate_idx < len(relation_candidate_ids):
                            original_relation_id = relation_candidate_ids[candidate_idx]
                            rerank_relation_ids.append(original_relation_id)
                except (ValueError, IndexError):
                    continue
            
            return rerank_relation_ids
            
        except Exception as e:
            print(f"Warning: LLM reranking failed ({e}), using original order")
            return relation_candidate_ids[:top_k]
    
    def _get_final_passages_from_relations(self, rerank_relation_ids: List[int], 
                                         final_top_k: int = 2) -> List[str]:
        """
        Get final passages from reranked relation IDs
        """
        final_passages = []
        final_passage_ids = []
        
        for relation_id in rerank_relation_ids:
            if relation_id < len(self.relations) and relation_id in self.relationid_2_passageids:
                for passage_id in self.relationid_2_passageids[relation_id]:
                    if passage_id not in final_passage_ids and passage_id < len(self.passages):
                        final_passage_ids.append(passage_id)
                        final_passages.append(self.passages[passage_id])
        return final_passages[:final_top_k]
    
    def _naive_rag_retrieval(self, query: str, top_k: int = 2) -> List[str]:
        """
        Naive RAG retrieval for comparison - direct similarity search on passages
        """
        query_embedding = embedding_model.embed_query(query)
        
        naive_passage_res = milvus_client.search(
            collection_name=self.passage_col_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text"],
        )[0]
        
        return [res["entity"]["text"] for res in naive_passage_res]
    
    def query_music_knowledge_with_llm_reranking(self, query: str, target_degree: int = 1,
                                               top_k: int = 3, final_top_k: int = 2,
                                               compare_with_naive: bool = True,
                                               entity_sim_thresh: float = 0.0,
                                               relation_sim_thresh: float = 0.0) -> str:
        """
        Complete Graph RAG with LLM reranking - the full pipeline
        """
        print(f"🕸️ Processing query with complete Graph RAG pipeline...")
        
        # Step 1: Extract entities and expand subgraph
        query_entities = self._extract_query_entities(query)
        print(f"📍 Extracted entities: {query_entities}")
        
        relation_candidate_texts, relation_candidate_ids = self._expand_subgraph(
            query_entities, query, target_degree, top_k, entity_sim_thresh, relation_sim_thresh
        )
        print(f"🔍 Found {len(relation_candidate_texts)} candidate relations from subgraph expansion")
        
        if not relation_candidate_texts:
            print("⚠️ No subgraph candidates found, falling back to naive RAG")
            naive_passages = self._naive_rag_retrieval(query, final_top_k)
            return self._generate_final_answer(query, naive_passages, "Naive RAG (fallback)")
        
        # Step 2: LLM Reranking - now with proper ID mapping!
        rerank_relation_ids = self._rerank_relations_with_llm(
            query, relation_candidate_texts, relation_candidate_ids, top_k
        )
        print(f"🧠 LLM reranked to {len(rerank_relation_ids)} most relevant relations")
        
        # Step 3: Get final passages from reranked relations
        passages_from_graph_rag = self._get_final_passages_from_relations(
            rerank_relation_ids, final_top_k
        )
        
        # Step 4: Compare with naive RAG if requested
        if compare_with_naive:
            passages_from_naive_rag = self._naive_rag_retrieval(query, final_top_k)
            
            print("\n" + "="*60)
            print("📊 COMPARISON: Graph RAG vs Naive RAG")
            print("="*60)
            print(f"📋 Passages from Naive RAG:")
            for i, passage in enumerate(passages_from_naive_rag, 1):
                print(f"  {i}. {passage[:150]}...")
            
            print(f"\n🕸️ Passages from Graph RAG:")
            for i, passage in enumerate(passages_from_graph_rag, 1):
                print(f"  {i}. {passage[:150]}...")
            print("="*60)
            
            # Generate answers from both methods
            answer_naive = self._generate_final_answer(query, passages_from_naive_rag, "Naive RAG")
            answer_graph = self._generate_final_answer(query, passages_from_graph_rag, "Graph RAG")
            
            return f"""🔍 **Naive RAG Answer:**
{answer_naive}

🕸️ **Graph RAG Answer:**
{answer_graph}

📈 **Analysis:** Graph RAG uses subgraph expansion and LLM reranking to find more contextually relevant passages, while Naive RAG relies purely on semantic similarity."""
        
        else:
            return self._generate_final_answer(query, passages_from_graph_rag, "Graph RAG")
    
    def _generate_final_answer(self, query: str, passages: List[str], method_name: str) -> str:
        """Generate final answer from retrieved passages"""
        context = "\n\n".join(passages)
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "human",
                """Use the following pieces of retrieved context about musical preferences and listening habits to answer the question. If there is not enough information in the retrieved context to answer the question, just say that you don't know.

Question: {question}
Context: {context}
Answer:""",
            )
        ])
        
        rag_chain = prompt | llm | StrOutputParser()
        
        answer = rag_chain.invoke({
            "question": query,
            "context": context
        })
        
        return f"[{method_name}] {answer}"
    
    def query_music_knowledge_advanced(self, query: str, target_degree: int = 1, 
                                     top_k: int = 3) -> str:
        """
        Advanced query using subgraph expansion with NER and adjacency matrices
        """
        print(f"Processing advanced query with subgraph expansion...")
        
        # Extract entities from query using simple NER
        query_entities = self._extract_query_entities(query)
        print(f"Extracted entities: {query_entities}")
        
        # Expand subgraph to get candidate relations
        relation_candidate_texts, relation_candidate_ids = self._expand_subgraph(
            query_entities, query, target_degree, top_k
        )
        
        print(f"Found {len(relation_candidate_texts)} candidate relations from subgraph expansion")
        
        # If we have candidate relations, use them; otherwise fall back to simple search
        if relation_candidate_texts:
            # Get embeddings for candidate relations and find most relevant
            query_embedding = embedding_model.embed_query(query)
            candidate_embeddings = embedding_model.embed_documents(relation_candidate_texts)
            
            # Calculate similarities
            similarities = []
            for i, candidate_emb in enumerate(candidate_embeddings):
                similarity = np.dot(query_embedding, candidate_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(candidate_emb)
                )
                similarities.append((similarity, relation_candidate_texts[i]))
            
            # Sort by similarity and take top results
            similarities.sort(reverse=True, key=lambda x: x[0])
            top_relations = [rel for _, rel in similarities[:top_k*2]]
            
            context = "Relevant musical relationships from subgraph expansion:\n"
            context += "\n".join([f"- {rel}" for rel in top_relations])
            
        else:
            # Fall back to simple similarity search
            print("No subgraph candidates found, falling back to similarity search")
            return self.query_music_knowledge(query, top_k)
        
        # Generate response using LLM
        prompt = f"""Based on the following musical relationships discovered through graph analysis, answer the user's question:

{context}

Question: {query}

Please provide a comprehensive answer based on the musical relationships and connections found. Focus on the specific relationships and connections that are most relevant to the question."""
        
        response = llm.invoke(prompt)
        return response.content
    
    def query_music_knowledge(self, query: str, top_k: int = 5) -> str:
        """Query the music knowledge graph using separate collections"""
        # Get query embedding
        query_embedding = embedding_model.embed_query(query)
        
        # Search across all three collections
        entity_results = milvus_client.search(
            collection_name=self.entity_col_name,
            data=[query_embedding],
            limit=top_k//3 + 1,
            output_fields=["text"]
        )
        
        relation_results = milvus_client.search(
            collection_name=self.relation_col_name,
            data=[query_embedding],
            limit=top_k//3 + 1,
            output_fields=["text"]
        )
        
        passage_results = milvus_client.search(
            collection_name=self.passage_col_name,
            data=[query_embedding],
            limit=top_k//3 + 1,
            output_fields=["text"]
        )
        
        # Extract relevant information from all collections
        context_parts = []
        
        # Add entity results
        for result in entity_results[0]:
            text = result["entity"]["text"]
            score = result["distance"]
            context_parts.append(f"(entity, relevance: {score:.3f}): {text}")
        
        # Add relation results
        for result in relation_results[0]:
            text = result["entity"]["text"]
            score = result["distance"]
            context_parts.append(f"(relation, relevance: {score:.3f}): {text}")
        
        # Add passage results
        for result in passage_results[0]:
            text = result["entity"]["text"]
            score = result["distance"]
            context_parts.append(f"(passage, relevance: {score:.3f}): {text}")
        
        context = "\n".join(context_parts)
        
        # Generate response using LLM
        prompt = f"""Based on the following music knowledge graph information, answer the user's question about their musical interests and collection:

Context from music knowledge graph:
{context}

Question: {query}

Please provide a comprehensive answer based on the musical information available. Use the entities (artists, genres, etc.), relations (connections between them), and passages (detailed descriptions) to give insights about the user's musical taste and collection."""
        
        response = llm.invoke(prompt)
        return response.content
    
    def save_graph_data(self, filepath: str = "music_graph_data.json"):
        """Save the graph data to a file for later use"""
        graph_data = {
            "entities": self.entities,
            "relations": self.relations,
            "passages": self.passages,
            "entityid_2_relationids": dict(self.entityid_2_relationids),
            "relationid_2_passageids": dict(self.relationid_2_passageids),
            "music_dataset": self.music_dataset
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"Graph data saved to {filepath}")

# Example usage
if __name__ == "__main__":
    # Example music directories - replace with your actual paths
    music_dirs = [
        "/Users/akritiwork/Music/iTunes/iTunes Media/Music",
        "/Users/akritiwork/Music/Personal Collection",
        # Add more directories as needed
    ]
    
    # Initialize the system
    music_rag = MusicGraphRAG(music_dirs)
    
    # Build the graph structure
    music_rag.build_graph_structure()
    
    # Create embeddings and store in Milvus
    music_rag.create_embeddings_and_store()
    
    # Save graph data
    music_rag.save_graph_data()
    
    # Example queries
    queries = [
        "What genres of music do I listen to?",
        "Which artists have the most albums in my collection?",
        "Tell me about my jazz music collection",
        "What albums do I have by [specific artist]?",
        "What's the diversity of my music taste?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print(f"Answer: {music_rag.query_music_knowledge(query)}")
        print("-" * 50)
