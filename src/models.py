import uuid
from typing import List, Optional
from pydantic import BaseModel, Field

class Publication(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    original_id: Optional[str]
    title: str
    authors: List[str]
    url: str
    pdf_url: Optional[str]
    abstract: Optional[str]
    source: str
