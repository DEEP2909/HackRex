"""
Embedding and semantic search engine
"""

import os
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
from datetime import datetime

# ML libraries
from sentence_transformers import SentenceTransformer
import faiss
import pinecone

from ..models.document import ProcessedChunk, Domain, DocumentType
from ..utils.logger import get_logger
from config.settings import Settings

logger = get_logger(__name__)

class EmbeddingSearchEngine:
    """Handles embeddings generation and semantic search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = None
        self.embedding_dim = 384  # Default for MiniLM
        
        # FAISS index
        self.faiss_index = None
        self.chunk_metadata = []
        self.doc_id_to_chunks = {}  # Map doc_id to chunk indices
        
        # Pinecone setup
        self.use_pinecone = False
        self.pinecone_index = None
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_size_limit = 10000
        
    async def initialize(self):
        """Initialize the embedding engine"""
        try:
            logger.info(f"Initializing embedding engine with model: {self.model_name}")
            
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            # Initialize FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            
            # Try to initialize Pinecone
            await self._init_pinecone()
            
            # Load existing index if available
            await self._load_existing_index()
            
            logger.info(f"Embedding engine initialized successfully. Dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding engine: {str(e)}")
            raise
    
    async def _init_pinecone(self):
        """Initialize Pinecone vector database"""
        try:
            settings = Settings()
            if settings.pinecone_api_key:
                pinecone.init(
                    api_key=settings.pinecone_api_key,
                    environment=settings.pinecone_env
                )
                
                index_name = "document-retrieval"
                
                # Create index if it doesn't exist
                if index_name not in pinecone.list_indexes():
                    pinecone.create_index(
                        index_name,
                        dimension=self.embedding_dim,
                        metric="cosine"
                    )
                    logger.info(f"Created Pinecone index: {index_name}")
                
                self.pinecone_index = pinecone.Index(index_name)
                self.use_pinecone = True
                logger.info("Pinecone initialized successfully")
                
        except Exception as e:
            logger.warning(f"Pinecone initialization failed: {e}. Using local FAISS only.")
            self.use_pinecone = False
    
    async def _load_existing_index(self):
        """Load existing FAISS index and metadata"""
        try:
            index_path = "./data/faiss_index.bin"
            metadata_path = "./data/chunk_metadata.pkl"
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # Load FAISS index
                self.faiss_index = faiss.read_index(index_path)
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.chunk_metadata = saved_data.get('chunks', [])
                    self.doc_id_to_chunks = saved_data.get('doc_mapping', {})
                
                logger.info(f"Loaded existing index with {len(self.chunk_metadata)} chunks")
                
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}. Starting fresh.")
    
    async def save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            os.makedirs("./data", exist_ok=True)
            
            # Save FAISS index
            index_path = "./data/faiss_index.bin"
            faiss.write_index(self.faiss_index, index_path)
            
            # Save metadata
            metadata_path = "./data/chunk_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunk_metadata,
                    'doc_mapping': self.doc_id_to_chunks
                }, f)
            
            logger.info("Index saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        try:
            if not texts:
                return np.array([])
            
            # Check cache first
            cache_keys = [hash(text) for text in texts]
            cached_embeddings = []
            texts_to_process = []
            cache_indices = []
            
            for i, (text, key) in enumerate(zip(texts, cache_keys)):
                if key in self.embedding_cache:
                    cached_embeddings.append((i, self.embedding_cache[key]))
                else:
                    texts_to_process.append(text)
                    cache_indices.append((i, key))
            
            # Generate embeddings for uncached texts
            new_embeddings = []
            if texts_to_process:
                embeddings = self.embedding_model.encode(
                    texts_to_process,
                    convert_to_numpy=True,
                    show_progress_bar=len(texts_to_process) > 10
                )
                
                # Normalize for cosine similarity
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                new_embeddings = embeddings
                
                # Cache new embeddings
                for (original_idx, cache_key), embedding in zip(cache_indices, embeddings):
                    self.embedding_cache[cache_key] = embedding
                    
                    # Limit cache size
                    if len(self.embedding_cache) > self.cache_size_limit:
                        # Remove oldest entries (simple FIFO)
                        oldest_key = next(iter(self.embedding_cache))
                        del self.embedding_cache[oldest_key]
            
            # Combine cached and new embeddings
            all_embeddings = np.zeros((len(texts), self.embedding_dim))
            
            # Insert cached embeddings
            for idx, embedding in cached_embeddings:
                all_embeddings[idx] = embedding
            
            # Insert new embeddings
            new_idx = 0
            for idx, _ in cache_indices:
                all_embeddings[idx] = new_embeddings[new_idx]
                new_idx += 1
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def index_chunks(self, chunks: List[ProcessedChunk]):
        """Index document chunks for retrieval"""
        try:
            if not chunks:
                return
            
            logger.info(f"Indexing {len(chunks)} chunks")
            
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.generate_embeddings(texts)
            
            if embeddings.size == 0:
                logger.warning("No embeddings generated")
                return
            
            # Add to FAISS index
            start_idx = self.faiss_index.ntotal
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Store metadata and update mappings
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i].tolist()
                chunk_dict = chunk.to_dict()
                
                # Add FAISS index information
                chunk_dict['faiss_idx'] = start_idx + i
                chunk_dict['indexed_at'] = datetime.now().isoformat()
                
                self.chunk_metadata.append(chunk_dict)
                
                # Update doc_id mapping
                doc_id = chunk.doc_id
                if doc_id not in self.doc_id_to_chunks:
                    self.doc_id_to_chunks[doc_id] = []
                self.doc_id_to_chunks[doc_id].append(len(self.chunk_metadata) - 1)
            
            # Add to Pinecone if available
            if self.use_pinecone:
                await self._index_in_pinecone(chunks, embeddings)
            
            # Save index periodically
            if len(self.chunk_metadata) % 100 == 0:
                await self.save_index()
            
            logger.info(f"Successfully indexed {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error indexing chunks: {str(e)}")
            raise
    
    async def _index_in_pinecone(self, chunks: List[ProcessedChunk], embeddings: np.ndarray):
        """Index chunks in Pinecone"""
        try:
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                metadata = {
                    'doc_id': chunk.doc_id,
                    'content': chunk.content[:1000],  # Pinecone metadata size limit
                    'chunk_index': chunk.chunk_index,
                    'domain': chunk.metadata.get('domain', 'general'),
                    'filename': chunk.metadata.get('filename', ''),
                    'doc_type': chunk.metadata.get('doc_type', ''),
                    'indexed_at': datetime.now().isoformat()
                }
                
                vectors.append((
                    chunk.chunk_id,
                    embedding.tolist(),
                    metadata
                ))
            
            # Batch upsert to Pinecone
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.pinecone_index.upsert(vectors=batch)
            
            logger.info(f"Indexed {len(vectors)} vectors in Pinecone")
            
        except Exception as e:
            logger.error(f"Error indexing in Pinecone: {str(e)}")
            # Don't raise - continue with local indexing
    
    async def search_similar_chunks(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.5,
        domain_filter: Optional[Domain] = None,
        doc_type_filter: Optional[List[DocumentType]] = None,
        doc_ids: Optional[List[str]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar chunks"""
        try:
            if not query.strip():
                return []
            
            # Generate query embedding
            query_embedding = await self.generate_embeddings([query])
            if query_embedding.size == 0:
                return []
            
            query_vector = query_embedding[0].astype('float32')
            
            # Search in FAISS
            scores, indices = self.faiss_index.search(
                query_vector.reshape(1, -1), 
                min(top_k * 2, self.faiss_index.ntotal)  # Get more results for filtering
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunk_metadata):  # Valid index
                    chunk_data = self.chunk_metadata[idx].copy()
                    similarity_score = float(score)
                    
                    # Apply threshold filter
                    if similarity_score < similarity_threshold:
                        continue
                    
                    # Apply domain filter
                    if domain_filter and chunk_data.get('metadata', {}).get('domain') != domain_filter.value:
                        continue
                    
                    # Apply doc type filter
                    if doc_type_filter:
                        chunk_doc_type = chunk_data.get('metadata', {}).get('doc_type')
                        if chunk_doc_type not in [dt.value for dt in doc_type_filter]:
                            continue
                    
                    # Apply doc_ids filter
                    if doc_ids and chunk_data.get('doc_id') not in doc_ids:
                        continue
                    
                    results.append((chunk_data, similarity_score))
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k results
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    async def search_by_doc_id(self, doc_id: str, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search within a specific document"""
        try:
            if doc_id not in self.doc_id_to_chunks:
                return []
            
            chunk_indices = self.doc_id_to_chunks[doc_id]
            
            # Generate query embedding
            query_embedding = await self.generate_embeddings([query])
            if query_embedding.size == 0:
                return []
            
            query_vector = query_embedding[0]
            
            # Calculate similarities for chunks in this document
            results = []
            for chunk_idx in chunk_indices:
                if chunk_idx < len(self.chunk_metadata):
                    chunk_data = self.chunk_metadata[chunk_idx]
                    chunk_embedding = np.array(chunk_data['embedding'])
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_vector, chunk_embedding)
                    results.append((chunk_data, float(similarity)))
            
            # Sort and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching by doc_id: {str(e)}")
            return []
    
    async def remove_document_chunks(self, doc_id: str):
        """Remove all chunks for a document"""
        try:
            if doc_id not in self.doc_id_to_chunks:
                logger.warning(f"Document {doc_id} not found in index")
                return
            
            chunk_indices = self.doc_id_to_chunks[doc_id]
            
            # Mark chunks as deleted (FAISS doesn't support deletion, so we mark them)
            for chunk_idx in chunk_indices:
                if chunk_idx < len(self.chunk_metadata):
                    self.chunk_metadata[chunk_idx]['deleted'] = True
            
            # Remove from doc mapping
            del self.doc_id_to_chunks[doc_id]
            
            # Remove from Pinecone if available
            if self.use_pinecone:
                chunk_ids = [self.chunk_metadata[idx]['chunk_id'] for idx in chunk_indices 
                           if idx < len(self.chunk_metadata)]
                if chunk_ids:
                    self.pinecone_index.delete(ids=chunk_ids)
            
            logger.info(f"Removed {len(chunk_indices)} chunks for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error removing document chunks: {str(e)}")
    
    async def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        try:
            if doc_id not in self.doc_id_to_chunks:
                return []
            
            chunk_indices = self.doc_id_to_chunks[doc_id]
            chunks = []
            
            for chunk_idx in chunk_indices:
                if chunk_idx < len(self.chunk_metadata):
                    chunk_data = self.chunk_metadata[chunk_idx]
                    if not chunk_data.get('deleted', False):
                        chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            return []
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            active_chunks = sum(1 for chunk in self.chunk_metadata if not chunk.get('deleted', False))
            
            # Domain distribution
            domain_counts = {}
            doc_type_counts = {}
            
            for chunk in self.chunk_metadata:
                if not chunk.get('deleted', False):
                    domain = chunk.get('metadata', {}).get('domain', 'unknown')
                    doc_type = chunk.get('metadata', {}).get('doc_type', 'unknown')
                    
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
            
            return {
                'total_chunks': len(self.chunk_metadata),
                'active_chunks': active_chunks,
                'deleted_chunks': len(self.chunk_metadata) - active_chunks,
                'total_documents': len(self.doc_id_to_chunks),
                'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0,
                'embedding_dimension': self.embedding_dim,
                'model_name': self.model_name,
                'using_pinecone': self.use_pinecone,
                'cache_size': len(self.embedding_cache),
                'domain_distribution': domain_counts,
                'doc_type_distribution': doc_type_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Save index before cleanup
            await self.save_index()
            logger.info("Embedding engine cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    async def rebuild_index(self):
        """Rebuild the entire index (useful for maintenance)"""
        try:
            logger.info("Rebuilding index...")
            
            # Get all active chunks
            active_chunks = [chunk for chunk in self.chunk_metadata if not chunk.get('deleted', False)]
            
            # Reset index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.chunk_metadata = []
            self.doc_id_to_chunks = {}
            
            # Re-index chunks in batches
            batch_size = 100
            for i in range(0, len(active_chunks), batch_size):
                batch_chunks = active_chunks[i:i + batch_size]
                
                # Convert back to ProcessedChunk objects
                processed_chunks = []
                for chunk_data in batch_chunks:
                    processed_chunk = ProcessedChunk(
                        chunk_id=chunk_data['chunk_id'],
                        doc_id=chunk_data['doc_id'],
                        content=chunk_data['content'],
                        embedding=chunk_data['embedding'],
                        chunk_index=chunk_data['chunk_index'],
                        metadata=chunk_data.get('metadata', {})
                    )
                    processed_chunks.append(processed_chunk)
                
                # Re-index batch
                await self.index_chunks(processed_chunks)
            
            await self.save_index()
            logger.info("Index rebuild completed")
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")
            raise
