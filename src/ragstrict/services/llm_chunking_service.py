"""LLM-assisted chunking service with unified API configuration"""

import aiohttp
import json
import re
from typing import List
from ragstrict.core.config import get_config


class LLMChunkingService:
    """Intelligent chunking service using LLM or simple rules"""
    
    def __init__(self):
        config = get_config()
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
    
    async def chunk_text(self, text: str, filename: str = "") -> List[str]:
        """
        Intelligently chunk text
        
        Tries LLM-based chunking if API enabled, falls back to simple chunking
        """
        
        if self.config.enable_api and self.config.llm_api_url:
            try:
                return await self._chunk_via_llm(text, filename)
            except Exception as e:
                print(f"⚠️  LLM chunking failed: {e}, falling back to simple chunking")
        
        return self._simple_chunk(text)
    
    async def _chunk_via_llm(self, text: str, filename: str = "") -> List[str]:
        """Chunk via LLM API"""
        
        if not self.config.llm_api_url:
            raise ValueError("LLM API URL not configured")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.config.llm_api_key or "",
        }
        
        # Build LLM request
        system_prompt = f"""You are a text chunking assistant. Split the text into semantically complete paragraphs, each around {self.chunk_size} characters.
Maintain semantic integrity and split at natural boundaries. Return JSON format: {{"chunks": ["paragraph1", "paragraph2", ...]}}"""

        user_prompt = f"""Filename: {filename}

Please chunk the following text:
{text[:2000]}"""  # Limit input length

        payload = {
            "model": self.config.llm_api_model or "qwen3-32b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 4000,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.llm_api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"LLM API error {response.status}: {error_text}")
                
                result = await response.json()
                
                # Parse response: {"choices": [{"message": {"content": "..."}}]}
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    
                    # Try parse JSON
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        if "chunks" in parsed:
                            return parsed["chunks"]
                
                # Failed to parse, raise error to trigger fallback
                raise Exception("Failed to parse LLM response")
    
    def _simple_chunk(self, text: str) -> List[str]:
        """Simple rule-based chunking - split by lines and size"""
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            
            if current_size + line_size > self.chunk_size and current_chunk:
                # Current chunk full, save and start new chunk
                chunks.append('\n'.join(current_chunk))
                
                # Overlap: keep last few lines
                overlap_lines = int(self.chunk_overlap / (current_size / len(current_chunk)) if current_chunk else 0)
                current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
                current_size = sum(len(l) for l in current_chunk)
            
            current_chunk.append(line)
            current_size += line_size
        
        # Save last chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks if chunks else [text]
