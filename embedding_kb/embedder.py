from typing import List, Optional
import re
from openai import OpenAI

class MemoryEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        """Create an embedding for a single text string"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a batch of texts"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]

class RecursiveTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text recursively using separators"""
        # Handle base cases
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                chunks = []
                splits = text.split(separator)
                current_chunk = []
                current_length = 0

                for split in splits:
                    split_length = len(split)

                    if current_length + split_length <= self.chunk_size:
                        current_chunk.append(split)
                        current_length += split_length + len(separator)
                    else:
                        # Join current chunks
                        if current_chunk:
                            chunks.append(separator.join(current_chunk))
                        
                        # Start new chunk
                        current_chunk = [split]
                        current_length = split_length

                # Add remaining chunk
                if current_chunk:
                    chunks.append(separator.join(current_chunk))

                # Handle any chunks that are still too large
                final_chunks = []
                for chunk in chunks:
                    if len(chunk) > self.chunk_size:
                        final_chunks.extend(self._split_by_chunk(chunk))
                    else:
                        final_chunks.append(chunk)

                return final_chunks

        # Fallback to splitting by character count
        return self._split_by_chunk(text)

    def _split_by_chunk(self, text: str) -> List[str]:
        """Split text by chunk size as a last resort"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap  # Apply overlap
        return chunks
