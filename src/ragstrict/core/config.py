"""Configuration management for RAGStrict"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    """RAGStrict configuration"""
    
    # Paths
    data_dir: Path
    db_path: Path
    models_dir: Path
    uploads_dir: Path
    
    # Database
    database_url: str
    
    # Embedding
    embedding_local_model: str
    embedding_dimension: int
    chunk_size: int
    chunk_overlap: int
    
    # MCP Server
    mcp_port: int
    mcp_host: str
    
    # Environment
    environment: str
    debug: bool
    offline_mode: bool
    
    # API Configuration
    enable_api: bool
    
    # Embedding API
    embedding_api_url: Optional[str]
    embedding_api_model: Optional[str]
    embedding_api_key: Optional[str]
    
    # LLM API
    llm_api_url: Optional[str]
    llm_api_model: Optional[str]
    llm_api_key: Optional[str]
    
    @classmethod
    def from_env(cls, config_dir: Optional[Path] = None) -> "Config":
        """Load configuration from environment variables"""
        
        if config_dir is None:
            config_dir = Path.cwd() / ".ragstrict"
        
        # Load .env file if exists (basic configuration)
        env_file = config_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        
        # Load .env.api file if exists (API configuration)
        api_env_file = config_dir / ".env.api"
        if api_env_file.exists():
            load_dotenv(api_env_file, override=True)
        
        # Create directories
        config_dir.mkdir(parents=True, exist_ok=True)
        data_dir = config_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        models_dir = config_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        uploads_dir = config_dir / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        # Database
        db_path = data_dir / "ragstrict.db"
        database_url = f"sqlite+aiosqlite:///{db_path}"
        
        # Embedding
        embedding_local_model = os.getenv(
            "EMBEDDING_LOCAL_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        
        # MCP
        mcp_port = int(os.getenv("MCP_PORT", "3000"))
        mcp_host = os.getenv("MCP_HOST", "localhost")

        # Environment
        environment = os.getenv("ENVIRONMENT", "development")
        debug = os.getenv("DEBUG", "false").lower() == "true"
        offline_mode = os.getenv("OFFLINE_MODE", "false").lower() == "true"
        
        # API Configuration
        enable_api = os.getenv("ENABLE_API", "false").lower() == "true"
        
        # Embedding API
        embedding_api_url = os.getenv("EMBEDDING_API_URL")
        embedding_api_model = os.getenv("EMBEDDING_API_MODEL", "Qwen3-Embedding-8B")
        embedding_api_key = os.getenv("EMBEDDING_API_KEY")
        
        # LLM API
        llm_api_url = os.getenv("LLM_API_URL")
        llm_api_model = os.getenv("LLM_API_MODEL", "qwen3-32b")
        llm_api_key = os.getenv("LLM_API_KEY")

        return cls(
            data_dir=data_dir,
            db_path=db_path,
            models_dir=models_dir,
            uploads_dir=uploads_dir,
            database_url=database_url,
            embedding_local_model=embedding_local_model,
            embedding_dimension=embedding_dimension,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            mcp_port=mcp_port,
            mcp_host=mcp_host,
            environment=environment,
            debug=debug,
            offline_mode=offline_mode,
            enable_api=enable_api,
            embedding_api_url=embedding_api_url,
            embedding_api_model=embedding_api_model,
            embedding_api_key=embedding_api_key,
            llm_api_url=llm_api_url,
            llm_api_model=llm_api_model,
            llm_api_key=llm_api_key,
        )


# Global config instance
_config: Optional[Config] = None


def get_config(config_dir: Optional[Path] = None) -> Config:
    """Get or create global config instance"""
    global _config
    if _config is None:
        _config = Config.from_env(config_dir)
    return _config


def reset_config():
    """Reset global config (for testing)"""
    global _config
    _config = None
