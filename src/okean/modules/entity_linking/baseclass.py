from abc import ABC
from typing import List, Dict
from dataclasses import dataclass
from okean.data_types.basic_types import Passage


@dataclass
class EntityLinkingResponse:
    passages: List[Passage]
    runtimes: Dict[str, float]


class EntityLinking(ABC):
    pass