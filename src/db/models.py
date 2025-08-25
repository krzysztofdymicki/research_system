from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Publication(BaseModel):
    id: Optional[str] = None
    search_result_id: Optional[int] = None
    original_id: Optional[str] = None
    title: str
    authors: List[str] = Field(default_factory=list)
    url: str
    pdf_url: Optional[str] = None
    abstract: Optional[str] = None
    source: str
    pdf_path: Optional[str] = None
    markdown: Optional[str] = None
    extractions_json: Optional[str] = None
    relevance_score: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Search(BaseModel):
    id: Optional[str] = None
    query: str
    sources: List[str] = Field(default_factory=list)
    max_results_per_source: int = 5
    created_at: Optional[datetime] = None


class RawSearchResult(BaseModel):
    id: Optional[int] = None
    search_id: str
    source: str
    original_id: Optional[str] = None
    title: str
    authors: List[str] = Field(default_factory=list)
    url: str
    pdf_url: Optional[str] = None
    abstract: Optional[str] = None
    relevance_score: Optional[float] = None
    analysis_json: Optional[str] = None
    analyzed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
