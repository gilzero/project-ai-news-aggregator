from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from dateutil import parser
import logging

logger = logging.getLogger(__name__)

class Article(BaseModel):
    """Data class for storing article information"""
    article_id: Optional[str] = None
    title: str = 'N/A'
    article_url: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    image_url: Optional[str] = None
    posted_time: Optional[str] = None
    relative_time: str = 'N/A'
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    scraped_at: str = Field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    posted_time_iso: Optional[str] = None

    @field_validator('article_url', 'source_url', 'image_url', mode='before')
    def validate_urls(cls, v):
        """Validate and clean URLs"""
        if v == 'N/A':
            return None
        return v

    @field_validator('posted_time', mode='before')
    def convert_posted_time(cls, v):
        """Convert posted_time to ISO format"""
        if v:
            try:
                dt = parser.parse(v)
                return dt.isoformat()
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse posted_time: {v}, error: {e}")
                return None
        return v 