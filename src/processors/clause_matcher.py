"""
Clause Matching Engine for identifying semantically similar clauses.
"""
import asyncio
from typing import List, Dict, Any, Tuple

from .embedding_engine import EmbeddingSearchEngine
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ClauseMatchingEngine:
    """
    Identifies relevant clauses from document chunks using semantic similarity.
    """
    def __init__(self, embedding_engine: EmbeddingSearchEngine):
        """
        Initializes the ClauseMatchingEngine.

        Args:
            embedding_engine: An instance of the embedding and search engine.
        """
        self.embedding_engine = embedding_engine
        logger.info("ClauseMatchingEngine initialized.")

    async def match_clauses(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        domain: str,
        top_k: int = 5,
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Matches a query against a list of chunks to find the best matching clauses.

        Args:
            query (str): The query or clause text to match.
            context_chunks (List[Dict[str, Any]]): A list of chunks to search within.
            domain (str): The domain (e.g., 'legal', 'insurance').
            top_k (int): The number of top clauses to return.
            threshold (float): The minimum similarity score to consider a match.

        Returns:
            List[Dict[str, Any]]: A list of matching chunks, sorted by relevance.
        """
        if not context_chunks:
            return []

        logger.info(f"Attempting to match clauses for query: '{query[:50]}...' in domain '{domain}'")

        try:
            # Generate embedding for the input query
            query_embedding = await self.embedding_engine.generate_embeddings([query])
            if query_embedding.size == 0:
                logger.warning("Could not generate embedding for clause matching query.")
                return []

            query_vector = query_embedding[0]

            # Generate embeddings for all context chunks
            chunk_contents = [chunk.get('content', '') for chunk in context_chunks]
            chunk_embeddings = await self.embedding_engine.generate_embeddings(chunk_contents)

            if chunk_embeddings.size == 0:
                logger.warning("Could not generate embeddings for context chunks.")
                return []
            
            # Calculate cosine similarity scores
            # Note: embeddings are already normalized in the engine
            import numpy as np
            similarity_scores = np.dot(chunk_embeddings, query_vector.T)

            # Combine chunks with their scores
            scored_chunks = []
            for i, chunk in enumerate(context_chunks):
                score = float(similarity_scores[i])
                if score >= threshold:
                    chunk_with_score = chunk.copy()
                    chunk_with_score['clause_match_score'] = score
                    scored_chunks.append(chunk_with_score)

            # Sort by score in descending order
            scored_chunks.sort(key=lambda x: x['clause_match_score'], reverse=True)

            logger.info(f"Found {len(scored_chunks)} matching clauses above threshold {threshold}.")
            return scored_chunks[:top_k]

        except Exception as e:
            logger.error(f"An error occurred during clause matching: {e}")
            return []
