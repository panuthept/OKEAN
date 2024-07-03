import numpy as np
from torch import Tensor
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
# from huggingface_hub import hf_hub_download
from okean.data_types.basic_types import Passage
from okean.packages.bgem3_package.bge_m3 import BGEM3FlagModel
from okean.modules.retrieval.baseclass import Retrieval, RetrievalConfig


@dataclass
class BGEm3Config(RetrievalConfig):
    max_query_length: int = 8192
    max_passage_length: int = 8192
    pooling_method: str = "cls"
    normalize_embeddings: bool = True
    similarity_distance: Dict[str, str] = field(
        default_factory=lambda: {
            "dense": "dot",
            "sparse": "dot",
            "colbert": "einsum",
        }
    )


class BGEm3(Retrieval):
    @property
    def available_models(self):
        return [
            "BAAI/bge-m3",
            "BAAI/bge-m3-retromae",
            "BAAI/bge-m3-unsupervised",
        ]
    
    def _build_models(self):
        # Build model
        self.model = BGEM3FlagModel(
            self.config.pretrained_model_name,
            pooling_method=self.config.pooling_method,
            normalize_embeddings=self.config.normalize_embeddings,
            use_fp16=self.use_fp16,
            device=self.device,
        )
        self.tokenizer = self.model.tokenizer

    def _precompute_corpus_embeddings(self, batch_size: int = 8, verbose: bool = True) -> Tensor:
        # Compute embeddings
        return self.model.encode(
            [data["text"] for data in self.corpus_contents],
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
            verbose=verbose,
        )

    def _input_preprocessing(
            self,
            passages: List[Passage],
            **kwargs,
    ) -> List[List[str]]:
        return [[passage.text for passage in passages]]
    
    def _encode(
            self, 
            texts: List[List[str]],
            batch_size: int = 8,
            **kwargs,
    ) -> Dict[str, Any]:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
    
    def _retrieve_top_k(
            self,
            query_embeddings: Dict[str, Any], 
            corpus_embeddings: Dict[str, Any], 
            top_k: int = 10, 
            dense_weight: float = 0.4,
            sparse_weight: float = 0.2,
            colbert_weight: float = 0.4,
    ) -> Tuple[List[float], List[int]]:
        dense_scores = np.array(query_embeddings["dense_vecs"] @ corpus_embeddings["dense_vecs"].T)
        sparse_scores = np.array([[self.model.compute_lexical_matching_score(query_lexical_weights, corpus_lexical_weights) for corpus_lexical_weights in corpus_embeddings["lexical_weights"]] for query_lexical_weights in query_embeddings["lexical_weights"]])
        colbert_scores = np.array([[self.model.colbert_score(query_colbert_vecs, corpus_colbert_vecs) for corpus_colbert_vecs in corpus_embeddings["colbert_vecs"]] for query_colbert_vecs in query_embeddings["colbert_vecs"]])
        combined_scores = (dense_weight * dense_scores) + (sparse_weight * sparse_scores) + (colbert_weight * colbert_scores)
        # Get top-k scores and indices
        top_k_indices = np.argsort(combined_scores, axis=1)[:, ::-1][:, :top_k]
        top_k_scores = np.take_along_axis(combined_scores, top_k_indices, axis=1)
        return top_k_scores.tolist(), top_k_indices.tolist()
    
    @staticmethod
    def _load_from_hub(hub_path: str):
        # Load config
        config = BGEm3Config(
            pretrained_model_name=hub_path,
        )
        # Load model
        # path_to_models = hf_hub_download(hub_path, "pytorch_model.bin")
        path_to_models = None
        return config, path_to_models


if __name__ == "__main__":
    model = BGEm3.from_pretrained(
        model_name_or_path="BAAI/bge-m3",
        text_corpus_path="./data/text_corpus/test.jsonl",
        precomputed_text_corpus_path="./data/precomputed_text_corpus/test",
    )
    model.precompute_corpus(
        save_path="./data/precomputed_text_corpus/test", batch_size=8, verbose=True, overwrite=True
    )
    
    texts = [
        "That is a happy person",
        "I have a dog",
        "The weather is nice today",
    ]
    response = model(texts, batch_size=8)
    print(response.runtimes)
    for passage in response.passages:
        print(passage.text)
        print("-" * 100)
        for relavant_passage in passage.relevant_passages:
            print(f"  - {relavant_passage.text} ({relavant_passage.confident:.2f})")
        print("=" * 100)
    print("*" * 100)