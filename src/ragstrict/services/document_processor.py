"""Document processing service"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragstrict.core.database import Document, Chunk, VectorEmbedding
from ragstrict.services.embedding_service import EmbeddingService
from ragstrict.services.llm_chunking_service import LLMChunkingService


class DocumentProcessor:
    """Process documents: parse, chunk, vectorize"""
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        chunking_service: Optional[LLMChunkingService] = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunking_service = chunking_service or LLMChunkingService()
    
    async def add_document(
        self,
        session: AsyncSession,
        filepath: Path,
        project_id: Optional[int] = None,
        auto_process: bool = True,
    ) -> Document:
        """Add a document to the database"""
        
        # Calculate file hash
        file_hash = self._calculate_hash(filepath)
        
        # Check if already exists
        result = await session.execute(
            select(Document).where(Document.file_hash == file_hash)
        )
        existing_doc = result.scalar_one_or_none()
        if existing_doc:
            return existing_doc
        
        # Create document record
        doc = Document(
            filename=filepath.name,
            filepath=str(filepath.absolute()),
            file_type=filepath.suffix,
            file_size=filepath.stat().st_size,
            file_hash=file_hash,
            project_id=project_id,
            status="pending",
        )
        session.add(doc)
        await session.flush()
        
        if auto_process:
            # Parse and chunk
            chunks = await self._parse_and_chunk(filepath)
            await self._save_chunks(session, doc.id, chunks)
            
            # Vectorize
            await self._vectorize_chunks(session, doc.id)
            
            # Update status
            doc.status = "completed"
        
        await session.commit()
        return doc
    
    def _calculate_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    async def _parse_and_chunk(self, filepath: Path) -> List[str]:
        """Parse file and split into chunks"""
        
        # Simple text extraction
        try:
            content = filepath.read_text(encoding='utf-8')
        except Exception:
            try:
                content = filepath.read_text(encoding='gbk')
            except Exception:
                content = ""
        
        # Use LLM chunking service (supports both intranet and internet modes)
        chunks = await self.chunking_service.chunk_text(content, filepath.name)
        
        return chunks
    
    async def _save_chunks(
        self,
        session: AsyncSession,
        document_id: int,
        chunks: List[str],
    ) -> List[Chunk]:
        """Save chunks to database"""
        
        chunk_objects = []
        for i, content in enumerate(chunks):
            chunk = Chunk(
                document_id=document_id,
                content=content,
                chunk_index=i,
                chunk_metadata=json.dumps({"length": len(content)}),
            )
            session.add(chunk)
            chunk_objects.append(chunk)
        
        await session.flush()
        return chunk_objects
    
    async def _vectorize_chunks(self, session: AsyncSession, document_id: int):
        """Generate embeddings for all chunks of a document"""
        
        # Get chunks
        result = await session.execute(
            select(Chunk).where(Chunk.document_id == document_id)
        )
        chunks = result.scalars().all()
        
        if not chunks:
            return
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_service.encode(texts)
        
        # Save embeddings
        for chunk, embedding in zip(chunks, embeddings):
            # Check if already exists
            result = await session.execute(
                select(VectorEmbedding).where(VectorEmbedding.chunk_id == chunk.id)
            )
            existing = result.scalar_one_or_none()
            
            if not existing:
                vec_emb = VectorEmbedding(
                    chunk_id=chunk.id,
                    embedding=json.dumps(embedding.tolist()),
                )
                session.add(vec_emb)
        
        await session.flush()
    
    async def get_document_stats(self, session: AsyncSession) -> Dict[str, Any]:
        """Get document statistics"""
        
        # Count documents
        result = await session.execute(select(Document))
        documents = result.scalars().all()
        
        # Count chunks
        result = await session.execute(select(Chunk))
        chunks = result.scalars().all()
        
        # Count embeddings
        result = await session.execute(select(VectorEmbedding))
        embeddings = result.scalars().all()
        
        return {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "total_embeddings": len(embeddings),
            "vectorization_rate": (
                len(embeddings) / len(chunks) * 100 if chunks else 0
            ),
        }
