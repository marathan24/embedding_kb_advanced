from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any
from naptha_sdk.schemas import KBConfig

class InputSchema(BaseModel):
    func_name: Literal["init", "run_query", "add_data"]
    func_input_data: Optional[Dict[str, Any]] = None

class RetrieverConfig(BaseModel):
    type: str = "vector"
    field: str = "embedding" 
    k: int = 5