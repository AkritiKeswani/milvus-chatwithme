"""
Demo dataset based on visible Spotify interface
This creates a knowledge graph from the artists and relationships visible in the screenshot
"""

import numpy as np
from collections import defaultdict
from typing import List, Set
from scipy.sparse import csr_matrix
from config import milvus_client, llm, embedding_model

# Demo dataset based on your Spotify interface
spotify_demo_dataset = [
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
        "passage": "The Weeknd is a key artist in your collection, representing the darker, more atmospheric side of R&B and pop. His inclusion in your top artists shows appreciation for moody, production-heavy music that bridges R&B, pop, and electronic elements.",
        "triplets": [
            ["The Weeknd", "is in", "top artists"],
            ["The Weeknd", "represents", "atmospheric R&B"],
            ["The Weeknd", "represents", "dark pop"],
            ["The Weeknd", "bridges", "R&B and electronic"],
            ["The Weeknd", "is known for", "moody production"],
            ["The Weeknd", "influences", "contemporary R&B"],
        ],
    },
    {
        "passage": "Kendrick Lamar appears in your top artists, representing conscious hip-hop and lyrical complexity. His presence alongside Drake shows appreciation for different styles within hip-hop - from mainstream appeal to artistic depth and social commentary.",
        "triplets": [
            ["Kendrick Lamar", "is in", "top artists"],
            ["Kendrick Lamar", "represents", "conscious hip-hop"],
            ["Kendrick Lamar", "is known for", "lyrical complexity"],
            ["Kendrick Lamar", "provides", "social commentary"],
            ["Kendrick Lamar", "contrasts with", "Drake"],
            ["Kendrick Lamar", "represents", "artistic depth"],
        ],
    },
    {
        "passage": "Billie Eilish in your top artists represents alternative pop and Gen Z musical sensibilities. Her unique sound and aesthetic connect to the broader alternative and indie pop movement, showing your openness to innovative and unconventional pop music.",
        "triplets": [
            ["Billie Eilish", "is in", "top artists"],
            ["Billie Eilish", "represents", "alternative pop"],
            ["Billie Eilish", "represents", "Gen Z music"],
            ["Billie Eilish", "has", "unique aesthetic"],
            ["Billie Eilish", "connects to", "indie pop movement"],
            ["Billie Eilish", "represents", "innovative pop"],
        ],
    },
    {
        "passage": "Lady Gaga's presence in your top artists shows appreciation for pop artistry, theatrical performance, and genre versatility. Her ability to span pop, jazz, and rock demonstrates your taste for artists who push creative boundaries.",
        "triplets": [
            ["Lady Gaga", "is in", "top artists"],
            ["Lady Gaga", "represents", "pop artistry"],
            ["Lady Gaga", "is known for", "theatrical performance"],
            ["Lady Gaga", "spans", "multiple genres"],
            ["Lady Gaga", "spans", "pop and jazz"],
            ["Lady Gaga", "spans", "pop and rock"],
            ["Lady Gaga", "pushes", "creative boundaries"],
        ],
    },
    {
        "passage": "Your genre diversity spans Hip-Hop (Drake, Kendrick Lamar, Post Malone), Pop (Taylor Swift, Billie Eilish, Lady Gaga), R&B (SZA, The Weeknd), Classic Rock (Fleetwood Mac), and Country (Morgan Wallen). This shows a broad musical palate that appreciates both mainstream and alternative approaches within each genre.",
        "triplets": [
            ["Your taste", "includes", "Hip-Hop"],
            ["Your taste", "includes", "Pop"],
            ["Your taste", "includes", "R&B"],
            ["Your taste", "includes", "Classic Rock"],
            ["Your taste", "includes", "Country"],
            ["Your taste", "appreciates", "mainstream music"],
            ["Your taste", "appreciates", "alternative approaches"],
            ["Your taste", "shows", "broad musical palate"],
        ],
    },
    {
        "passage": "The radio stations you engage with show preference for artist-curated experiences, where you discover music through the lens of artists you already enjoy. This indicates you value musical curation and artist-to-artist connections in your discovery process.",
        "triplets": [
            ["You", "prefer", "artist-curated experiences"],
            ["You", "discover through", "favorite artists"],
            ["You", "value", "musical curation"],
            ["You", "appreciate", "artist connections"],
            ["Radio stations", "provide", "discovery method"],
            ["Your discovery", "is", "artist-driven"],
        ],
    },
]

class SpotifyDemoRAG:
    def __init__(self):
        self.entities = []
        self.relations = []
        self.passages = []
        self.entityid_2_relationids = defaultdict(list)
        self.relationid_2_passageids = defaultdict(list)
        
        # Adjacency matrices for subgraph expansion
        self.entity_relation_adj = None
        self.entity_adj_1_degree = None
        self.relation_adj_1_degree = None
        
    def build_graph_from_spotify_data(self):
        """Build graph structure from Spotify demo data"""
        print("Building graph from Spotify interface data...")
        
        for passage_id, dataset_info in enumerate(spotify_demo_dataset):
            passage, triplets = dataset_info["passage"], dataset_info["triplets"]
            self.passages.append(passage)
            
            for triplet in triplets:
                # Add entities
                if triplet[0] not in self.entities:
                    self.entities.append(triplet[0])
                if triplet[2] not in self.entities:
                    self.entities.append(triplet[2])
                
                # Create relation
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
        
        print(f"Built graph with {len(self.entities)} entities, {len(self.relations)} relations, {len(self.passages)} passages")
    
    def create_embeddings_and_store(self):
        """Create embeddings and store in Milvus following the proper pattern"""
        print("Creating embeddings for Spotify data...")
        
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
        
        print("Successfully stored Spotify embeddings in separate Milvus collections")
        
        # Build adjacency matrices for subgraph expansion
        self._build_adjacency_matrices()
    
    def _create_milvus_collections(self, embedding_dim: int):
        """Create separate Milvus collections for entities, relations, and passages"""
        self.entity_col_name = "spotify_entity_collection"
        self.relation_col_name = "spotify_relation_collection"  
        self.passage_col_name = "spotify_passage_collection"
        
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
        
        print(f"Created Spotify collections: {self.entity_col_name}, {self.relation_col_name}, {self.passage_col_name}")
    
    def _milvus_insert(self, collection_name: str, text_list):
        """Insert data into Milvus collection with proper batching"""
        from tqdm import tqdm
        
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
        print("Building adjacency matrices for Spotify subgraph expansion...")
        
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
        
        print("Spotify adjacency matrices built successfully")
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract music entities from query for Spotify data"""
        query_lower = query.lower()
        query_entities = []
        
        # Check for entities in the query
        for entity in self.entities:
            if entity.lower() in query_lower:
                query_entities.append(entity)
        
        # If no entities found, use keyword extraction
        if not query_entities:
            music_keywords = ['artist', 'song', 'album', 'genre', 'music', 'taste', 'style']
            words = query_lower.split()
            
            for word in words:
                for entity in self.entities:
                    if word in entity.lower() and entity not in query_entities:
                        query_entities.append(entity)
                        break
        
        return query_entities[:3]
    
    def query_spotify_knowledge_advanced(self, query: str, target_degree: int = 1, 
                                       top_k: int = 3) -> str:
        """
        Advanced Spotify query using subgraph expansion
        """
        print(f"Processing advanced Spotify query with subgraph expansion...")
        
        # Extract entities from query
        query_entities = self._extract_query_entities(query)
        print(f"Extracted Spotify entities: {query_entities}")
        
        # Expand subgraph if we have adjacency matrices
        if self.entity_relation_adj is not None:
            relation_candidates = self._expand_spotify_subgraph(
                query_entities, query, target_degree, top_k
            )
            
            print(f"Found {len(relation_candidates)} candidate relations from Spotify subgraph")
            
            if relation_candidates:
                # Rank candidates by similarity to query
                query_embedding = embedding_model.embed_query(query)
                candidate_embeddings = embedding_model.embed_documents(relation_candidates)
                
                similarities = []
                for i, candidate_emb in enumerate(candidate_embeddings):
                    similarity = np.dot(query_embedding, candidate_emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(candidate_emb)
                    )
                    similarities.append((similarity, relation_candidates[i]))
                
                similarities.sort(reverse=True, key=lambda x: x[0])
                top_relations = [rel for _, rel in similarities[:top_k*2]]
                
                context = "Spotify musical relationships from subgraph analysis:\n"
                context += "\n".join([f"- {rel}" for rel in top_relations])
                
                prompt = f"""Based on the following musical relationships from your Spotify listening patterns, answer the question:

{context}

Question: {query}

Provide insights about your musical taste and listening habits based on these discovered relationships."""
                
                response = llm.invoke(prompt)
                return response.content
        
        # Fall back to simple search
        return self.query_spotify_knowledge(query, top_k)
    
    def _expand_spotify_subgraph(self, query_entities: List[str], query: str, 
                               target_degree: int = 1, top_k: int = 3) -> List[str]:
        """Expand Spotify subgraph using adjacency matrices"""
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

        entity_relation_adj_target_degree = entity_adj_target_degree @ self.entity_relation_adj
        
        # Expand relations
        expanded_relations_from_relation = set()
        filtered_hit_relation_ids = [
            relation_res["entity"]["id"] for relation_res in relation_search_res
        ]
        
        for hit_relation_id in filtered_hit_relation_ids:
            if hit_relation_id < relation_adj_target_degree.shape[0]:
                expanded_relations_from_relation.update(
                    relation_adj_target_degree[hit_relation_id].nonzero()[1].tolist()
                )
        
        expanded_relations_from_entity = set()
        filtered_hit_entity_ids = []
        
        for one_entity_search_res in entity_search_res:
            for one_entity_res in one_entity_search_res:
                filtered_hit_entity_ids.append(one_entity_res["entity"]["id"])
        
        for filtered_hit_entity_id in filtered_hit_entity_ids:
            if filtered_hit_entity_id < entity_relation_adj_target_degree.shape[0]:
                expanded_relations_from_entity.update(
                    entity_relation_adj_target_degree[filtered_hit_entity_id].nonzero()[1].tolist()
                )
        
        # Merge expanded relations
        relation_candidate_ids = list(
            expanded_relations_from_relation | expanded_relations_from_entity
        )
        
        # Get relation texts
        relation_candidate_texts = [
            self.relations[relation_id] for relation_id in relation_candidate_ids
            if relation_id < len(self.relations)
        ]
        
        return relation_candidate_texts
    
    def query_spotify_knowledge(self, query: str, top_k: int = 5) -> str:
        """Query the Spotify music knowledge using separate collections"""
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
        
        prompt = f"""Based on the following information about the user's Spotify listening habits and musical preferences, answer their question:

Context from Spotify knowledge graph:
{context}

Question: {query}

Please provide insights about their musical taste, preferences, and listening patterns based on the available information. Use the entities (artists, genres, etc.), relations (connections between them), and passages (detailed descriptions) to give comprehensive insights about the user's musical preferences."""
        
        response = llm.invoke(prompt)
        return response.content

# Demo usage
if __name__ == "__main__":
    print("🎵 Initializing Spotify Demo RAG...")
    spotify_rag = SpotifyDemoRAG()
    
    # Build graph
    spotify_rag.build_graph_from_spotify_data()
    
    # Create embeddings
    spotify_rag.create_embeddings_and_store()
    
    # Demo queries with both simple and advanced methods
    demo_queries = [
        "What genres of music do I listen to?",
        "Tell me about my hip-hop preferences", 
        "Which artists show my taste for mainstream vs alternative music?",
        "How does Post Malone connect to other artists in my taste?",
        "What does my artist selection say about my musical evolution?"
    ]
    
    print("\n" + "="*50)
    print("🎯 Demo Queries - Comparing Simple vs Advanced Graph RAG")
    print("="*50)
    
    for query in demo_queries:
        print(f"\n🎵 Query: {query}")
        
        print("\n📊 Simple Similarity Search:")
        simple_answer = spotify_rag.query_spotify_knowledge(query)
        print(f"{simple_answer}")
        
        print("\n🕸️ Advanced Subgraph Expansion:")
        advanced_answer = spotify_rag.query_spotify_knowledge_advanced(query)
        print(f"{advanced_answer}")
        
        print("-" * 60)
