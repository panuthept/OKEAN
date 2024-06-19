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
    confident: Optional[float] = None
    entities: Optional[Entity] = None
    candidates: Optional[List[Entity]] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"
    

@dataclass
class Passage(BaseDataType):
    text: str
    confident: Optional[float] = None
    entities: Optional[List[Span]] = None
    relevant_passages: Optional[List['Passage']] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"


@dataclass
class Document(BaseDataType):
    passages: List[Passage]
    confident: Optional[float] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"