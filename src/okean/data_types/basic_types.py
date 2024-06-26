from torch import FloatTensor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from okean.data_types.baseclass import BaseDataType


@dataclass
class Entity(BaseDataType):
    identifier: str
    confident: Optional[float|FloatTensor] = None
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"


@dataclass
class Span(BaseDataType):
    start: int
    end: int
    surface_form: str
    confident: Optional[float|FloatTensor] = None
    entity: Optional[Entity] = None
    candidates: Optional[List[Entity]] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"
    

@dataclass
class Passage(BaseDataType):
    text: str
    confident: Optional[float|FloatTensor] = None
    relevant_entities: Optional[List[Span]] = None
    relevant_passages: Optional[List['Passage']] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"


@dataclass
class Document(BaseDataType):
    passages: List[Passage]
    confident: Optional[float|FloatTensor] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"