import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Publication(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    original_id: Optional[str]
    title: str
    authors: List[str]
    url: str
    pdf_url: Optional[str]
    abstract: Optional[str]
    source: str


class Search(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    query: str
    sources: List[str] = Field(default_factory=list)
    max_results_per_source: int = 5
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RawSearchResult(BaseModel):
    id: Optional[int] = None
    search_id: uuid.UUID
    source: str
    original_id: Optional[str]
    title: str
    authors: List[str] = Field(default_factory=list)
    url: str
    pdf_url: Optional[str]
    abstract: Optional[str]
    relevance_score: Optional[float] = None
    analysis_json: Optional[str] = None
    analyzed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
