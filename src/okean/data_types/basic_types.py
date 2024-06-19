from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from okean.data_types.baseclass import BaseDataType


@dataclass
class Entity(BaseDataType):
    identifier: str
    confident: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"


@dataclass
class Span(BaseDataType):
    start: int
    end: int
    surface_form: str
    entities: Optional[Entity] = None
    candidates: Optional[List[Entity]] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"


@dataclass
class Doc(BaseDataType):
    text: str
    confident: Optional[float] = None
    entities: Optional[List[Span]] = None
    relevant_docs: Optional[List['Doc']] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"