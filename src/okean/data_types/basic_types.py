from torch import FloatTensor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from okean.data_types.baseclass import BaseDataType


@dataclass
class Entity(BaseDataType):
    identifier: str
    logit: Optional[float|FloatTensor] = None
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
    logit: Optional[float|FloatTensor] = None
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
    logit: Optional[float|FloatTensor] = None
    confident: Optional[float|FloatTensor] = None
    spans: Optional[List[Span]] = None
    relations: Optional[List[Dict[str, Any]]] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"


@dataclass
class Document(BaseDataType):
    passages: List[Passage]
    logit: Optional[float|FloatTensor] = None
    confident: Optional[float|FloatTensor] = None

    def __repr__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self) -> str:
        attributes = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in self.__dict__.items() if v is not None])
        return f"{self.__class__.__name__}({attributes})"