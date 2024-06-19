from abc import ABC
from typing import List, Optional
from okean.data_types.basic_types import Passage


class EntityLinking(ABC):
    def __call__(
            self, 
            texts: List[str]|str = None, 
            passages: List[Passage]|Passage = None,
    ) -> List[Passage]:
        # Cast `texts` to `passages` if `passages` is not provided
        if passages is None:
            assert texts is not None, "Either `text` or `passages` must be provided."
            if isinstance(texts, list):
                passages = [Passage(text=t) for t in texts]
            else:
                passages = [Passage(text=texts)]
        # Ensure that `passages` is a list of `Passage` objects
        if not isinstance(passages, list):
            passages = [passages]
        return passages