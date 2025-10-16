"""Database management for RAGStrict"""

import asyncio
import sys
from pathlib import Path
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.sql import func

from ragstrict.core.config import get_config


# Base class for all models
Base = declarative_base()


# Models (from backend/app/models/database.py)
class Document(Base):
    """Document model"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False, index=True)
    filepath = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String, unique=True, nullable=False, index=True)
    project_id = Column(Integer, nullable=True, index=True)
    classification_id = Column(Integer, nullable=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    status = Column(String, default="pending", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Chunk(Base):
    """Chunk model"""
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_metadata = Column(Text, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class VectorEmbedding(Base):
    """Vector embedding model"""
    __tablename__ = "vector_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, ForeignKey("chunks.id"), unique=True, nullable=False, index=True)
    embedding = Column(Text, nullable=False)  # JSON array
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Entity(Base):
    """Entity model"""
    __tablename__ = "entities"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    entity_type = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    properties = Column(Text, nullable=True)  # JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Relation(Base):
    """Relation model"""
    __tablename__ = "relations"
    
    id = Column(Integer, primary_key=True, index=True)
    source_entity_id = Column(Integer, ForeignKey("entities.id"), nullable=False, index=True)
    target_entity_id = Column(Integer, ForeignKey("entities.id"), nullable=False, index=True)
    relation_type = Column(String, nullable=False, index=True)
    properties = Column(Text, nullable=True)  # JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Integer, default=1)
    is_superuser = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Global engine and session maker
_engine = None
_async_session_maker = None


def init_database(config_dir: Path = None) -> None:
    """Initialize database connection"""
    global _engine, _async_session_maker
    
    config = get_config(config_dir)
    
    # Create async engine
    _engine = create_async_engine(
        config.database_url,
        echo=config.debug,
        future=True,
    )
    
    # Create session maker
    _async_session_maker = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def create_tables():
    """Create all tables"""
    global _engine
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables():
    """Drop all tables"""
    global _engine
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    global _async_session_maker
    if _async_session_maker is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with _async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def clean_database():
    """Clean all data from database"""
    from sqlalchemy import text
    async with get_db_session() as session:
        # Delete in order to respect foreign keys
        await session.execute(text("DELETE FROM relations"))
        await session.execute(text("DELETE FROM entities"))
        await session.execute(text("DELETE FROM vector_embeddings"))
        await session.execute(text("DELETE FROM chunks"))
        await session.execute(text("DELETE FROM documents"))
        await session.commit()


def run_async(coro):
    """Helper to run async function in sync context"""
    # Set Windows-specific event loop policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Check if there's a running event loop
    try:
        loop = asyncio.get_running_loop()
        # If we get here, there IS a running loop - shouldn't happen in CLI
        # Try to use it anyway
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop - this is the expected case
        return asyncio.run(coro)
