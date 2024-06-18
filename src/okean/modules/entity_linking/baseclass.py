from abc import ABC
from typing import List
from okean.data_types.basic_types import Doc


class BaseEntityLinking(ABC):
    def __call__(
            self, 
            texts: List[str]|str = None, 
            docs: List[Doc]|Doc = None,
    ) -> List[Doc]:
        pass