from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Baseclass:
    def from_dict(self, dict_data: Dict[str, Any]) -> "Baseclass":
        for key, value in dict_data.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class Entity(Baseclass):
    identifier: str
    confident: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Span(Baseclass):
    start: int
    end: int
    surface_form: str
    entity: Optional[List[Entity] | Entity] = None


@dataclass
class Doc(Baseclass):
    text: str
    entities: Optional[List[Span]] = None