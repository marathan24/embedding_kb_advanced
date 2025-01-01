from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any
from naptha_sdk.schemas import KBConfig

class InputSchema(BaseModel):
    function_name: Literal["init", "run_query", "add_data"]
    function_input_data: Optional[Dict[str, Any]] = None

class EmbedderConfig(BaseModel):
    model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = ["\n\n", "\n", ". ", " ", ""]
    embedding_dim: int = 1536

class RetrieverConfig(BaseModel):
    type: str = "vector"
    field: str = "embedding" 
    k: int = 5

class EmbeddingKBConfig(KBConfig):
    embedder: EmbedderConfig
    retriever: RetrieverConfig