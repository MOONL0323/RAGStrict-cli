"""Statistics service"""

from typing import Dict, Any
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ragstrict.core.database import Document, Chunk, VectorEmbedding, Entity, Relation


class StatsService:
    """Get statistics about the system"""
    
    async def get_stats(self, session: AsyncSession) -> Dict[str, Any]:
        """Get all statistics"""
        
        # Documents
        result = await session.execute(select(func.count(Document.id)))
        doc_count = result.scalar()
        
        # Chunks
        result = await session.execute(select(func.count(Chunk.id)))
        chunk_count = result.scalar()
        
        # Embeddings
        result = await session.execute(select(func.count(VectorEmbedding.id)))
        embedding_count = result.scalar()
        
        # Entities
        result = await session.execute(select(func.count(Entity.id)))
        entity_count = result.scalar()
        
        # Relations
        result = await session.execute(select(func.count(Relation.id)))
        relation_count = result.scalar()
        
        # Calculate rates
        vectorization_rate = (
            (embedding_count / chunk_count * 100) if chunk_count > 0 else 0.0
        )
        
        return {
            "documents": {
                "total": doc_count,
            },
            "chunks": {
                "total": chunk_count,
            },
            "embeddings": {
                "total": embedding_count,
                "vectorization_rate": round(vectorization_rate, 2),
            },
            "knowledge_graph": {
                "entities": entity_count,
                "relations": relation_count,
            },
        }
    
    async def get_document_list(
        self,
        session: AsyncSession,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        """Get list of documents"""
        
        result = await session.execute(
            select(Document)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        documents = result.scalars().all()
        
        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "status": doc.status,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
            }
            for doc in documents
        ]
