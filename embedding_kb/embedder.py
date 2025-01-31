# /embedding_kb/embedder.py

from typing import List, Optional
import os
import random
import numpy as np
from openai import OpenAI
import semchunk

class MemoryEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", chunk_size: int = 2000, chunk_overlap: int = 200):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = semchunk.chunkerify('gpt-4o', chunk_size=self.chunk_size)
    
    def embed_text(self, text: str) -> List[float]:
        """Create an embedding for a single text string"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a batch of texts"""
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

class SemChunkTextSplitter:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = semchunk.chunkerify('gpt-4o', chunk_size=self.chunk_size)

    def split_text(self, text: str) -> List[str]:
        """Split text using semchunk"""
        # Adjusted to handle the correct number of return values
        chunks, offsets = self.chunker(text, offsets=True)
        return chunks
