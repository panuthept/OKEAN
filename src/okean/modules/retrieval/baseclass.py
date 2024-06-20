import os
import shutil
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from okean.data_types.basic_types import Passage
from usearch.index import Index, BatchMatches, Matches, search


@dataclass
class IndexConfig:
    ndim: int
    metric: str = "ip"
    dtype: str = "f32"
    connectivity: int = 16
    expansion_add: int = 128
    expansion_search: int = 64
    multi: bool = False

    def to_dict(self):
        return self.__dict__


class Retriever(ABC):
    @abstractmethod
    def save_corpus(self, corpus_path: str):
        raise NotImplementedError
    
    @abstractmethod
    def load_corpus(self, corpus_path: str):
        raise NotImplementedError
    
    @abstractmethod
    def corpus_encoding(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def queries_encoding(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def build_corpus(self, corpus_path: str, texts: List[str], batch_size: int = 8, remove_existing: bool = False, skip_existing: bool = True):
        raise NotImplementedError
    
    @abstractmethod
    def __call__(
            self, 
            texts: List[str]|str = None, 
            passages: List[Passage]|Passage = None,
            batch_size: int = 8,
            k: int = 10,
    ) -> List[Passage]:
        raise NotImplementedError


class DenseRetriever(Retriever):
    def __init__(self, index_config: IndexConfig, corpus_path: Optional[str] = None):
        self.index_config = index_config

        self.corpus_contents = []
        self.corpus_embeddings = None
        self.index = None

        if corpus_path is not None and os.path.exists(corpus_path):
            print(f"Loading corpus from {corpus_path}.")
            self.load_corpus(corpus_path)

    def save_corpus(self, corpus_path: str):
        os.makedirs(corpus_path, exist_ok=True)
        np.save(os.path.join(corpus_path, "embeddings.npy"), self.corpus_embeddings)
        with open(os.path.join(corpus_path, "contents.txt"), "w") as f:
            f.write("\n".join(self.corpus_contents))

    def load_corpus(self, corpus_path: str):
        self.corpus_embeddings = np.load(os.path.join(corpus_path, "embeddings.npy"))
        with open(os.path.join(corpus_path, "contents.txt"), "r") as f:
            self.corpus_contents = f.read().splitlines()

        self.index = Index(**self.index_config.to_dict())
        self.index.add(np.arange(len(self.corpus_contents)), self.corpus_embeddings)

    def build_corpus(
            self, 
            corpus_path: str, 
            texts: List[str], 
            batch_size: int = 8, 
            remove_existing: bool = False, 
            skip_existing: bool = True
    ):
        if os.path.exists(corpus_path) and not remove_existing:
            if skip_existing:
                print(f"Corpus already exists at {corpus_path}. Set `remove_existing=True` to overwrite.")
                return
            raise FileExistsError(f"Corpus already exists at {corpus_path}. Set `skip_existing=True` to skip or `remove_existing=True` to overwrite.")
        
        if os.path.exists(corpus_path) and remove_existing:
            print(f"Removing existing corpus at {corpus_path}.")
            self.corpus_contents = []
            self.corpus_embeddings = None
            self.index = None
            shutil.rmtree(corpus_path)

        self.corpus_contents = texts
        self.corpus_embeddings = self.corpus_encoding(texts, batch_size=batch_size)

        self.index = Index(**self.index_config.to_dict())
        self.index.add(np.arange(len(self.corpus_contents)), self.corpus_embeddings)
        self.save_corpus(corpus_path)

    def __call__(
            self, 
            texts: List[str]|str = None, 
            passages: List[Passage]|Passage = None,
            batch_size: int = 8,
            k: int = 10,
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

        if self.corpus_embeddings is None:
            raise ValueError("Corpus not built. Use `build_corpus` method to build the corpus.")

        queries: List[str] = [passage.text for passage in passages]
        queries_embeddings = self.queries_encoding(queries, batch_size=batch_size)

        k = min(k, len(self.corpus_contents))
        matches: Union[BatchMatches, Matches] = self.index.search(queries_embeddings, count=k, exact=(len(self.corpus_contents) < 100000))
        if isinstance(matches, Matches):
            matches = [matches]

        passages = deepcopy(passages)
        for passage, match in zip(passages, matches):
            passage.relations = [{"relevant_passage": [Passage(text=self.corpus_contents[idx], confident=score) for score, idx in zip(match.distances, match.keys)]}]
        return passages