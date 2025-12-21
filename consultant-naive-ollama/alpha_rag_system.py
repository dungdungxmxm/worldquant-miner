"""
RAG (Retrieval-Augmented Generation) System for Alpha Generation

This module implements a RAG system to improve alpha generation by:
1. Learning from successful historical alphas
2. Retrieving similar successful patterns
3. Providing context to the AI model for better generation

Supports two backends:
- Qdrant Vector Database (recommended for production)
- TF-IDF (fallback for development/testing)

Author: AI Assistant
Date: 2025-12-19
Priority: [CAO] - High impact on alpha quality
"""

import json
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

# Step 1: Try to import Qdrant dependencies
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("Qdrant dependencies not available. Falling back to TF-IDF.")

# Step 2: Import TF-IDF as fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaRAGSystem:
    """
    RAG system for alpha generation.

    Supports two backends:
    - Qdrant Vector Database (production-grade, persistent)
    - TF-IDF (fallback, in-memory only)
    """

    def __init__(self, hopeful_alphas_file: str = "hopeful_alphas.json",
                 submitted_alphas_file: str = "submission_log.json"):
        """
        Initialize RAG system.

        Args:
            hopeful_alphas_file: Path to hopeful alphas JSON file
            submitted_alphas_file: Path to submitted alphas log file
        """
        self.hopeful_alphas_file = hopeful_alphas_file
        self.submitted_alphas_file = submitted_alphas_file
        self.alpha_database: List[Dict] = []

        # Counter for tracking embedding cache cleanup
        # Cleanup is triggered every 10 alphas to prevent VRAM leak
        self.alpha_add_count = 0

        # Step 1: Determine which backend to use
        use_qdrant = os.getenv('USE_QDRANT', 'false').lower() == 'true'
        self.use_qdrant = use_qdrant and QDRANT_AVAILABLE

        if use_qdrant and not QDRANT_AVAILABLE:
            logger.warning("USE_QDRANT=true but Qdrant dependencies not available. Falling back to TF-IDF.")

        # Step 2: Initialize backend-specific components
        if self.use_qdrant:
            self._init_qdrant_backend()
        else:
            self._init_tfidf_backend()

        # Step 3: Load existing alphas
        self.load_alpha_database()

    def _init_qdrant_backend(self) -> None:
        """Initialize Qdrant vector database backend."""
        logger.info("Initializing Qdrant backend for RAG system...")

        # Step 1: Get Qdrant connection settings from environment
        qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))

        # Step 2: Initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            logger.warning("Falling back to TF-IDF backend")
            self.use_qdrant = False
            self._init_tfidf_backend()
            return

        # Step 3: Initialize embedding model
        # Using all-MiniLM-L6-v2: Fast, lightweight, good for short texts
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.vector_size = 384  # all-MiniLM-L6-v2 produces 384-dim vectors
            logger.info("Loaded sentence-transformers/all-MiniLM-L6-v2 embedding model")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.warning("Falling back to TF-IDF backend")
            self.use_qdrant = False
            self._init_tfidf_backend()
            return

        # Step 4: Create or verify collection
        self.collection_name = "alphas"
        self._ensure_collection_exists()

    def _init_tfidf_backend(self) -> None:
        """Initialize TF-IDF fallback backend."""
        logger.info("Initializing TF-IDF backend for RAG system...")

        # Step 1: Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer='char',  # Character-level for alpha expressions
            ngram_range=(2, 5),  # 2-5 character n-grams
            max_features=1000
        )
        self.alpha_vectors = None

    def _ensure_collection_exists(self) -> None:
        """Ensure Qdrant collection exists with correct schema."""
        try:
            # Step 1: Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name in collection_names:
                logger.info(f"Qdrant collection '{self.collection_name}' already exists")
                return

            # Step 2: Create collection if it doesn't exist
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE  # Cosine similarity for semantic search
                )
            )
            logger.info(f"Created Qdrant collection '{self.collection_name}' with vector size {self.vector_size}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def load_alpha_database(self) -> None:
        """Load successful alphas from files into database."""
        logger.info("Loading alpha database for RAG system...")
        
        # Load hopeful alphas
        if os.path.exists(self.hopeful_alphas_file):
            try:
                with open(self.hopeful_alphas_file, 'r') as f:
                    hopeful_alphas = json.load(f)
                    self.alpha_database.extend(hopeful_alphas)
                    logger.info(f"Loaded {len(hopeful_alphas)} hopeful alphas")
            except Exception as e:
                logger.error(f"Error loading hopeful alphas: {e}")
        
        # Load submitted alphas
        if os.path.exists(self.submitted_alphas_file):
            try:
                with open(self.submitted_alphas_file, 'r') as f:
                    submission_log = json.load(f)
                    submitted = submission_log.get('submitted_alphas', [])
                    self.alpha_database.extend(submitted)
                    logger.info(f"Loaded {len(submitted)} submitted alphas")
            except Exception as e:
                logger.error(f"Error loading submitted alphas: {e}")
        
        # Build vector index if we have alphas
        if self.alpha_database:
            self.build_vector_index()
        else:
            logger.warning("No alphas found in database. RAG system will have limited functionality.")
    
    def build_vector_index(self) -> None:
        """Build vector index for alpha expressions (Qdrant or TF-IDF)."""
        logger.info("Building vector index for RAG system...")

        # Step 1: Extract expressions
        expressions = [alpha.get('expression', '') for alpha in self.alpha_database]

        # Step 2: Filter out empty expressions
        expressions = [expr for expr in expressions if expr]

        if not expressions:
            logger.warning("No valid expressions found for vectorization")
            return

        # Step 3: Build index based on backend
        if self.use_qdrant:
            self._build_qdrant_index(expressions)
        else:
            self._build_tfidf_index(expressions)

    def _build_qdrant_index(self, expressions: List[str]) -> None:
        """Build Qdrant vector index."""
        try:
            # Step 1: Generate embeddings for all expressions
            logger.info(f"Generating embeddings for {len(expressions)} expressions...")
            embeddings = self.embedding_model.encode(expressions, show_progress_bar=False)

            # Step 2: Prepare points for upsert
            points = []
            for idx, (alpha, embedding) in enumerate(zip(self.alpha_database, embeddings)):
                # Use expression hash as ID for deduplication
                point_id = hash(alpha.get('expression', '')) % (2**63)  # Ensure positive int64

                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        'expression': alpha.get('expression', ''),
                        'fitness': alpha.get('fitness', 0),
                        'sharpe': alpha.get('sharpe', 0),
                        'turnover': alpha.get('turnover', 0),
                        'timestamp': alpha.get('timestamp', 0),
                        'alpha_id': alpha.get('alpha_id', ''),
                        'returns': alpha.get('returns', 0),
                        'grade': alpha.get('grade', '')
                    }
                ))

            # Step 3: Upsert points to Qdrant (batch operation)
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"✅ Built Qdrant index with {len(points)} alphas")
        except Exception as e:
            logger.error(f"Error building Qdrant index: {e}")
            raise

    def _build_tfidf_index(self, expressions: List[str]) -> None:
        """Build TF-IDF vector index (fallback)."""
        try:
            self.alpha_vectors = self.vectorizer.fit_transform(expressions)
            logger.info(f"✅ Built TF-IDF index with {len(expressions)} alphas")
        except Exception as e:
            logger.error(f"Error building TF-IDF index: {e}")
    
    def retrieve_similar_alphas(self, query: str = None, top_k: int = 5,
                                min_fitness: float = 0.5) -> List[Dict]:
        """
        Retrieve similar successful alphas.

        Args:
            query: Query expression (if None, returns top alphas by fitness)
            top_k: Number of alphas to retrieve
            min_fitness: Minimum fitness threshold

        Returns:
            List of similar alpha dictionaries
        """
        if not self.alpha_database:
            logger.warning("Alpha database is empty")
            return []

        # Step 1: Route to appropriate backend
        if self.use_qdrant:
            return self._retrieve_qdrant(query, top_k, min_fitness)
        else:
            return self._retrieve_tfidf(query, top_k, min_fitness)

    def _retrieve_qdrant(self, query: str = None, top_k: int = 5,
                         min_fitness: float = 0.5) -> List[Dict]:
        """Retrieve similar alphas using Qdrant."""
        try:
            # Step 1: If no query, return top alphas by fitness
            if query is None:
                # Query Qdrant with filter for fitness
                from qdrant_client.models import Filter, FieldCondition, Range

                search_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="fitness",
                                range=Range(gte=min_fitness)
                            )
                        ]
                    ),
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False
                )

                # Extract and sort by fitness
                alphas = [point.payload for point in search_result[0]]
                alphas = sorted(alphas, key=lambda x: x.get('fitness', 0), reverse=True)
                return alphas[:top_k]

            # Step 2: Generate query embedding
            query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]

            # Step 3: Search Qdrant with filter
            from qdrant_client.models import Filter, FieldCondition, Range

            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="fitness",
                            range=Range(gte=min_fitness)
                        )
                    ]
                ),
                limit=top_k,
                with_payload=True
            )

            # Step 4: Format results
            similar_alphas = []
            for hit in search_result:
                alpha = hit.payload
                alpha['similarity_score'] = hit.score  # Cosine similarity score
                similar_alphas.append(alpha)

            return similar_alphas
        except Exception as e:
            logger.error(f"Error retrieving from Qdrant: {e}")
            return []

    def _retrieve_tfidf(self, query: str = None, top_k: int = 5,
                        min_fitness: float = 0.5) -> List[Dict]:
        """Retrieve similar alphas using TF-IDF (fallback)."""
        # Step 1: Filter by fitness
        filtered_alphas = [
            alpha for alpha in self.alpha_database
            if alpha.get('fitness', 0) >= min_fitness
        ]

        if not filtered_alphas:
            logger.warning(f"No alphas found with fitness >= {min_fitness}")
            return []

        # Step 2: If no query, return top alphas by fitness
        if query is None or self.alpha_vectors is None:
            sorted_alphas = sorted(
                filtered_alphas,
                key=lambda x: x.get('fitness', 0),
                reverse=True
            )
            return sorted_alphas[:top_k]

        # Step 3: Vectorize query
        try:
            query_vector = self.vectorizer.transform([query])
        except Exception as e:
            logger.error(f"Error vectorizing query: {e}")
            return []

        # Step 4: Calculate similarities
        similarities = cosine_similarity(query_vector, self.alpha_vectors)[0]

        # Step 5: Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Step 6: Return top-k alphas
        similar_alphas = [filtered_alphas[i] for i in top_indices if i < len(filtered_alphas)]

        # Step 7: Add similarity scores
        for i, alpha in enumerate(similar_alphas):
            alpha['similarity_score'] = float(similarities[top_indices[i]])

        return similar_alphas

    def get_successful_patterns(self, min_fitness: float = 0.7) -> Dict:
        """
        Extract common patterns from successful alphas.

        Args:
            min_fitness: Minimum fitness threshold for "successful"

        Returns:
            Dictionary with pattern statistics
        """
        successful_alphas = [
            alpha for alpha in self.alpha_database
            if alpha.get('fitness', 0) >= min_fitness
        ]

        if not successful_alphas:
            return {}

        # Extract patterns
        patterns = {
            'operators': {},
            'data_fields': {},
            'avg_fitness': 0,
            'avg_sharpe': 0,
            'avg_turnover': 0,
            'total_count': len(successful_alphas)
        }

        # Analyze operators and data fields
        for alpha in successful_alphas:
            expression = alpha.get('expression', '')

            # Extract operators (functions with parentheses)
            import re
            operators = re.findall(r'(\w+)\(', expression)
            for op in operators:
                patterns['operators'][op] = patterns['operators'].get(op, 0) + 1

            # Extract data fields (lowercase words not followed by parentheses)
            data_fields = re.findall(r'\b([a-z_][a-z0-9_]*)\b(?!\s*\()', expression)
            for field in data_fields:
                if field not in ['and', 'or', 'not', 'if', 'else']:  # Filter keywords
                    patterns['data_fields'][field] = patterns['data_fields'].get(field, 0) + 1

        # Calculate averages
        patterns['avg_fitness'] = np.mean([a.get('fitness', 0) for a in successful_alphas])
        patterns['avg_sharpe'] = np.mean([a.get('sharpe', 0) for a in successful_alphas if a.get('sharpe')])
        patterns['avg_turnover'] = np.mean([a.get('turnover', 0) for a in successful_alphas if a.get('turnover')])

        return patterns

    def generate_rag_context(self, top_k: int = 5, min_fitness: float = 0.6) -> str:
        """
        Generate RAG context for prompt enhancement.

        Args:
            top_k: Number of examples to include
            min_fitness: Minimum fitness threshold

        Returns:
            Formatted context string for prompt
        """
        # Get top successful alphas
        top_alphas = self.retrieve_similar_alphas(query=None, top_k=top_k, min_fitness=min_fitness)

        if not top_alphas:
            return ""

        # Get successful patterns
        patterns = self.get_successful_patterns(min_fitness=min_fitness)

        # Build context
        context_parts = []

        # Add successful patterns summary
        if patterns:
            context_parts.append("=== Successful Alpha Patterns ===")
            context_parts.append(f"Total successful alphas: {patterns.get('total_count', 0)}")
            context_parts.append(f"Average fitness: {patterns.get('avg_fitness', 0):.3f}")
            context_parts.append(f"Average Sharpe: {patterns.get('avg_sharpe', 0):.3f}")

            # Top operators
            if patterns.get('operators'):
                top_ops = sorted(patterns['operators'].items(), key=lambda x: x[1], reverse=True)[:10]
                context_parts.append("\nMost successful operators:")
                for op, count in top_ops:
                    context_parts.append(f"  - {op}: used {count} times")

            # Top data fields
            if patterns.get('data_fields'):
                top_fields = sorted(patterns['data_fields'].items(), key=lambda x: x[1], reverse=True)[:10]
                context_parts.append("\nMost successful data fields:")
                for field, count in top_fields:
                    context_parts.append(f"  - {field}: used {count} times")

        # Add example successful alphas
        context_parts.append("\n=== Example Successful Alphas ===")
        for i, alpha in enumerate(top_alphas, 1):
            context_parts.append(f"\nExample {i}:")
            context_parts.append(f"  Expression: {alpha.get('expression', 'N/A')}")
            context_parts.append(f"  Fitness: {alpha.get('fitness', 0):.3f}")
            context_parts.append(f"  Sharpe: {alpha.get('sharpe', 0):.3f}")
            context_parts.append(f"  Turnover: {alpha.get('turnover', 0):.3f}")

        return "\n".join(context_parts)

    def add_alpha_to_database(self, alpha: Dict) -> None:
        """
        Add a new alpha to the database and update index with duplicate prevention.

        Args:
            alpha: Alpha dictionary with expression and metrics
        """
        # Step 1: Check for duplicates by expression
        expression = alpha.get('expression', '')
        alpha_id = alpha.get('alpha_id', '')

        # Check if expression already exists in database
        for existing_alpha in self.alpha_database:
            if existing_alpha.get('expression') == expression:
                logger.info(f"Alpha expression already in database, skipping duplicate: {expression[:50]}...")
                return
            # Also check by alpha_id
            if alpha_id and existing_alpha.get('alpha_id') == alpha_id:
                logger.info(f"Alpha ID {alpha_id} already in database, skipping duplicate")
                return

        # Step 2: Add to in-memory database (no duplicates)
        self.alpha_database.append(alpha)
        logger.debug(f"Added alpha to in-memory database: {expression[:50]}...")

        # Step 3: Add to vector index based on backend
        if self.use_qdrant:
            self._add_to_qdrant(alpha)
        else:
            # For TF-IDF, rebuild index every 10 alphas
            if len(self.alpha_database) % 10 == 0:
                self.build_vector_index()
                logger.info(f"Updated TF-IDF index with {len(self.alpha_database)} alphas")

    def _add_to_qdrant(self, alpha: Dict) -> None:
        """
        Add single alpha to Qdrant index with automatic deduplication.

        Note: Qdrant's upsert operation automatically handles duplicates.
        If the same point_id (hash of expression) already exists, it will be updated
        instead of creating a duplicate entry.

        Args:
            alpha: Alpha dictionary with expression and metrics
        """
        try:
            # Step 1: Generate embedding
            expression = alpha.get('expression', '')
            if not expression:
                logger.warning("Cannot add alpha with empty expression")
                return

            embedding = self.embedding_model.encode([expression], show_progress_bar=False)[0]

            # Step 2: Create point with hash-based ID for automatic deduplication
            # Using hash(expression) ensures same expression = same ID = upsert instead of duplicate
            point_id = hash(expression) % (2**63)  # Ensure positive int64

            alpha_id = alpha.get('alpha_id', '')
            logger.debug(f"Creating Qdrant point with ID {point_id} for alpha {alpha_id}")

            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    'expression': expression,
                    'fitness': alpha.get('fitness', 0),
                    'sharpe': alpha.get('sharpe', 0),
                    'turnover': alpha.get('turnover', 0),
                    'timestamp': alpha.get('timestamp', 0),
                    'alpha_id': alpha_id,
                    'returns': alpha.get('returns', 0),
                    'grade': alpha.get('grade', '')
                }
            )

            # Step 3: Upsert to Qdrant (automatically handles duplicates)
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            logger.info(f"Upserted alpha to Qdrant (ID: {point_id}): {expression[:50]}...")

            # Step 4: Cleanup embedding cache periodically
            # Increment counter and cleanup every 10 alphas to prevent VRAM leak
            self.alpha_add_count += 1
            if self.alpha_add_count % 10 == 0:
                self.cleanup_embedding_cache()
                logger.debug(f"Triggered embedding cache cleanup after {self.alpha_add_count} alphas")
        except Exception as e:
            logger.error(f"Error adding alpha to Qdrant: {e}")

    def save_database(self) -> None:
        """Save the alpha database to file."""
        try:
            # Save to a separate RAG database file
            rag_db_file = "rag_alpha_database.json"
            with open(rag_db_file, 'w') as f:
                json.dump(self.alpha_database, f, indent=2)
            logger.info(f"Saved RAG database with {len(self.alpha_database)} alphas to {rag_db_file}")
        except Exception as e:
            logger.error(f"Error saving RAG database: {e}")

    def get_statistics(self) -> Dict:
        """Get RAG system statistics."""
        stats = {
            'backend': 'Qdrant' if self.use_qdrant else 'TF-IDF',
            'total_alphas': len(self.alpha_database),
            'hopeful_alphas': len([a for a in self.alpha_database if a.get('fitness', 0) > 0.5]),
            'high_quality_alphas': len([a for a in self.alpha_database if a.get('fitness', 0) > 0.7]),
            'avg_fitness': np.mean([a.get('fitness', 0) for a in self.alpha_database]) if self.alpha_database else 0
        }

        # Add backend-specific stats
        if self.use_qdrant:
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                stats['vector_index_size'] = collection_info.points_count
                stats['vector_dimension'] = self.vector_size
            except Exception as e:
                logger.error(f"Error getting Qdrant stats: {e}")
                stats['vector_index_size'] = 0
                stats['vector_dimension'] = self.vector_size
        else:
            stats['vector_index_size'] = self.alpha_vectors.shape if self.alpha_vectors is not None else (0, 0)

        return stats

    def cleanup_embedding_cache(self):
        """
        Cleanup Sentence Transformers embedding cache to free VRAM.

        This method should be called periodically to prevent VRAM leak
        from accumulated embeddings. The cleanup is particularly important
        for long-running processes that generate many embeddings.

        The method performs:
        1. Clear PyTorch CUDA cache used by Sentence Transformers
        2. Python garbage collection to free unused objects

        Note: This does NOT unload the embedding model from GPU, only clears
        the cached embeddings and intermediate tensors.
        """
        try:
            import gc
            import torch

            # Step 1: Clear any cached embeddings from CUDA memory
            # Check if embedding model has CUDA device before cleanup
            if hasattr(self.embedding_model, '_target_device'):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for all CUDA operations to complete

            # Step 2: Python garbage collection
            # Free unused Python objects and their associated memory
            gc.collect()

            logger.debug("🧹 Cleaned up Sentence Transformers embedding cache")
        except Exception as e:
            logger.warning(f"⚠️ Embedding cache cleanup failed: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = AlphaRAGSystem()

    # Get statistics
    stats = rag.get_statistics()
    print(f"\n=== RAG System Statistics ===")
    print(f"Total alphas: {stats['total_alphas']}")
    print(f"Hopeful alphas: {stats['hopeful_alphas']}")
    print(f"High quality alphas: {stats['high_quality_alphas']}")
    print(f"Average fitness: {stats['avg_fitness']:.3f}")

    # Generate context for prompt
    context = rag.generate_rag_context(top_k=5, min_fitness=0.6)
    print(f"\n=== RAG Context ===")
    print(context)

    # Retrieve similar alphas
    similar = rag.retrieve_similar_alphas(query="ts_mean(revenue, 20)", top_k=3)
    print(f"\n=== Similar Alphas ===")
    for alpha in similar:
        print(f"Expression: {alpha.get('expression')}")
        print(f"Fitness: {alpha.get('fitness'):.3f}")
        print(f"Similarity: {alpha.get('similarity_score', 0):.3f}")
        print()


