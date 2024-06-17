from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Entity:
    identifier: str
    confident: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Span:
    start: int
    end: int
    surface_form: str
    entity: Optional[List[Entity] | Entity] = None


@dataclass
class Doc:
    text: str
    entities: Optional[List[Span]] = None