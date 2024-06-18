from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from okean.data_types.baseclass import BaseDataType


@dataclass
class Entity(BaseDataType):
    identifier: str
    confident: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Span(BaseDataType):
    start: int
    end: int
    surface_form: str
    entity: Optional[Entity] = None
    candidates: Optional[List[Entity]] = None


@dataclass
class Doc(BaseDataType):
    text: str
    entities: Optional[List[Span]] = None