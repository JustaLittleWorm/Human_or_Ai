from pydantic import BaseModel, Field
from typing import List, Dict

class PredictRequest(BaseModel):
    text: str = Field(min_length=30, description="Abstract/text to analyze")

class ModelResult(BaseModel):
    name: str
    ai: float
    human: float

class PredictResponse(BaseModel):
    input_chars: int
    models: List[ModelResult]
    avg: Dict[str, float]
