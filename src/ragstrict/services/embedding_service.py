"""Embedding service with unified API configuration and fallback"""

import asyncio
from typing import List
import numpy as np
import aiohttp

from ragstrict.core.config import get_config


class EmbeddingService:
    """Generate embeddings using API or local model with fallback"""
    
    def __init__(self):
        config = get_config()
        self.config = config
        self.dimension = config.embedding_dimension
        self._model = None
    
    def _load_local_model(self):
        """Lazy load local sentence-transformers model"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.config.embedding_local_model,
                local_files_only=self.config.offline_mode
            )
        return self._model
    
    def _encode_local(self, texts: List[str]) -> np.ndarray:
        """Encode using local model (synchronous)"""
        model = self._load_local_model()
        return model.encode(texts, convert_to_numpy=True)
    
    async def _encode_via_api(self, texts: List[str]) -> np.ndarray:
        """Encode via API endpoint"""
        if not self.config.embedding_api_url:
            raise ValueError("Embedding API URL not configured")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.config.embedding_api_key or "",
        }
        
        payload = {
            "input": texts,
            "model": self.config.embedding_api_model or "Qwen3-Embedding-8B",
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.embedding_api_url,
                headers=headers,
                json=payload,
                timeout=timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Embedding API error {response.status}: {error_text}")
                
                result = await response.json()
                
                # Parse response: {"data": [{"embedding": [...]}, ...]}
                if "data" in result:
                    embeddings = [item["embedding"] for item in result["data"]]
                    return np.array(embeddings)
                else:
                    raise Exception(f"Unexpected API response format: {result}")
    
    async def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings with API + fallback"""
        
        # Try API if enabled
        if self.config.enable_api and self.config.embedding_api_url:
            try:
                return await self._encode_via_api(texts)
            except Exception as e:
                # Log error and fallback
                print(f"⚠️  Embedding API failed: {e}, falling back to local model")
        
        # Use local model (fallback or default)
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._encode_local(texts)
        )
        return embeddings
    
    async def encode_single(self, text: str) -> np.ndarray:
        """Encode single text"""
        result = await self.encode([text])
        return result[0]
