"""Vector search service"""

import json
from typing import List, Dict, Any, Tuple
import numpy as np

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragstrict.core.database import Chunk, VectorEmbedding, Document
from ragstrict.services.embedding_service import EmbeddingService


class VectorSearch:
    """Semantic search using vector embeddings"""
    
    def __init__(self, embedding_service: EmbeddingService = None):
        self.embedding_service = embedding_service or EmbeddingService()
    
    async def search(
        self,
        session: AsyncSession,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        
        # Generate query embedding
        query_embedding = await self.embedding_service.encode_single(query)
        
        # Get all embeddings
        result = await session.execute(
            select(VectorEmbedding, Chunk, Document)
            .join(Chunk, VectorEmbedding.chunk_id == Chunk.id)
            .join(Document, Chunk.document_id == Document.id)
        )
        rows = result.all()
        
        if not rows:
            return []
        
        # Calculate similarities
        similarities: List[Tuple[float, Dict[str, Any]]] = []
        
        for vec_emb, chunk, doc in rows:
            # Parse embedding
            chunk_embedding = np.array(json.loads(vec_emb.embedding))
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, chunk_embedding)
            
            if similarity >= min_score:
                similarities.append((
                    similarity,
                    {
                        "chunk_id": chunk.id,
                        "document_id": doc.id,
                        "document_name": doc.filename,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "similarity": float(similarity),
                    }
                ))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        return [item[1] for item in similarities[:top_k]]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
