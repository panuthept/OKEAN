import os
import faiss
import shutil
import numpy as np
from copy import deepcopy
from typing import List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from okean.data_types.basic_types import Passage


@dataclass
class FaissEngineConfig:
    index_type: str = "Flat"
    distance: str = "IP"
    dim: int = 768
    lsh_nbits_factor: int = 4
    hnsw_m: int = 64
    hnsw_ef_search: int = 32
    hnsw_ef_construction: int = 64
    ivf_nlist: int = 128
    ivf_nprobe: int = 8


class FaissEngine:
    def __init__(self, config: FaissEngineConfig):
        assert config.index_type in ["Flat", "LSH", "HNSW", "IVF"], "Invalid index type."
        assert config.distance in ["IP", "L2"], "Invalid distance type."

        if config.index_type == "LSH" and config.dim > 128:
            print(f"LSH index is not recommended for dim > 128. Got dim={config.dim}. Consider using HNSW index instead.")

        self.config = config
        self.index = self._create_index(config)

    @staticmethod
    def _create_index(config: FaissEngineConfig) -> faiss.Index:
        if config.index_type == "Flat":
            index = faiss.IndexFlatIP(config.dim) if config.distance == "IP" else faiss.IndexFlatL2(config.dim)
        elif config.index_type == "LSH":
            index = faiss.IndexLSH(config.dim, config.dim * config.lsh_nbits_factor)
        elif config.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(config.dim, config.hnsw_m)
            index.hnsw.efSearch = config.hnsw_ef_search
            index.hnsw.efConstruction = config.hnsw_ef_construction
        elif config.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(config.dim) if config.distance == "IP" else faiss.IndexFlatL2(config.dim)
            index = faiss.IndexIVFFlat(quantizer, config.dim, config.ivf_nlist)
        return index
    
    def add(self, vectors: np.ndarray):
        if self.config.index_type == "IVF":
            self.index.train(vectors)
            self.index.add(vectors)
            self.index.nprobe = self.config.ivf_nprobe
        else:
            self.index.add(vectors)
    
    def search(self, query: np.ndarray, k: int = 10) -> List[int]:
        scores, indices = self.index.search(query, k)
        return scores, indices


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
    def __init__(self, search_config: Optional[FaissEngineConfig] = None, corpus_path: Optional[str] = None):
        self.search_config = search_config if search_config is not None else FaissEngineConfig()

        self.corpus_contents = []
        self.corpus_embeddings = None
        self.search_engine = None

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
        self.search_engine = FaissEngine(self.search_config)
        self.search_engine.add(self.corpus_embeddings)

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
            shutil.rmtree(corpus_path)

        self.corpus_contents = texts
        self.corpus_embeddings = self.corpus_encoding(texts, batch_size=batch_size)

        self.search_engine = FaissEngine(self.search_config)
        self.search_engine.add(self.corpus_embeddings)
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
        scoress, indicess = self.search_engine.search(queries_embeddings, k=min(k, len(self.corpus_contents)))

        passages = deepcopy(passages)
        for passage, scores, indices in zip(passages, scoress, indicess):
            passage.relations = [{"relevant_passage": Passage(text=self.corpus_contents[idx], confident=score) for score, idx in zip(scores, indices)}]
        return passages