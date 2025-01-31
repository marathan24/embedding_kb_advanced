# /embedding_kb/schemas.py

from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List

class InputSchema(BaseModel):
    func_name: Literal["init", "run_query", "add_data"]
    func_input_data: Optional[Dict[str, Any]] = None

class RetrieverConfig(BaseModel):
    type: str = "vector"
    field: str = "embedding" 
    k: int = 5

class LLMConfigOptions(BaseModel):
    chunk_size: int = 2000
    chunk_overlap: int = 200
    separators: List[str] = ["\n\n", "\n", ". ", " ", ""]
    embedding_dim: int = 1536

class LLMConfig(BaseModel):
    config_name: str
    client: str
    model: str
    options: LLMConfigOptions