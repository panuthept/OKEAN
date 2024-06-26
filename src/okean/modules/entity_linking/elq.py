import os
import json
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from okean.data_types.config_types import IndexConfig
from usearch.index import Index, BatchMatches, Matches
from okean.data_types.basic_types import Passage, Span, Entity
from okean.modules.entity_linking.baseclass import EntityLinking
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from okean.packages.elq_package.biencoder.biencoder import BiEncoderRanker
from okean.packages.elq_package.biencoder.data_process import get_candidate_representation


@dataclass
class ELQConfig:
    lowercase: bool = True
    bert_model: str = "bert-large-uncased"
    path_to_model: Optional[str] = None
    load_cand_enc_only: bool = False
    max_context_length: int = 128
    max_cand_length: int = 128
    out_dim: int = 1
    pull_from_layer: int = -1
    add_linear: bool = False
    data_parallel: bool = False
    no_cuda: bool = False

    def to_dict(self):
        return self.__dict__


class ELQ(EntityLinking):
    def __init__(
            self, 
            config: ELQConfig,
            entity_corpus_path: str,
            max_candidates: int = 30,
            index_config: Optional[IndexConfig] = None, 
            precomputed_entity_corpus_path: Optional[str] = None,
            device: Optional[str] = None,
            use_fp16: bool = True,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.max_candidates = max_candidates

        self.model = BiEncoderRanker(config.to_dict())
        self.model.to(self.device)
        self.model.eval()
        if use_fp16: self.model.model.half()
        self.tokenizer = self.model.tokenizer

        # self.embeddings_dim = self.model.model.context_encoder.bert_model.config.hidden_size

        if index_config is None: index_config = IndexConfig(ndim=1024, metric="ip", dtype="f32")
        self.index_config = index_config

        self.corpus_contents = self._load_entity_corpus(entity_corpus_path)
        self.corpus_embeddings = None
        self.index = None
        if precomputed_entity_corpus_path:
            self._load_precomputed_entity_corpus(precomputed_entity_corpus_path)

    @staticmethod
    def _load_entity_corpus(load_path: str) -> List[Dict[str, Any]]:
        corpus_contents = []
        with open(load_path, "r") as f:
            for line in f:
                entity = json.loads(line)
                corpus_contents.append(entity)
        return corpus_contents
    
    def _load_precomputed_entity_corpus(self, load_path: str):
        self.corpus_embeddings = np.load(os.path.join(load_path, "embeddings.npy"))

        self.index = Index(**self.index_config.to_dict())
        self.index.add(np.arange(len(self.corpus_contents)), self.corpus_embeddings)

    def _tokenize_entity_corpus(self, verbose: bool = True):
        titles = [entity["name"] for entity in self.corpus_contents]
        descs = [entity["desc"] for entity in self.corpus_contents]

        tokenized_entity_corpus = []
        for title, desc in tqdm(zip(titles, descs), total=len(titles), desc="Tokenizing entities", disable=not verbose):
            tokenized_entity = get_candidate_representation(
                desc, self.tokenizer, self.config.max_cand_length, title
            )["ids"][0]
            tokenized_entity_corpus.append(tokenized_entity)
        return tokenized_entity_corpus
    
    def precompute_entity_corpus_tokens(self, save_path: str, verbose: bool = True):
        tokenized_entity_corpus = self._tokenize_entity_corpus(verbose=verbose)

        os.makedirs(save_path, exist_ok=True)
        torch.save(tokenized_entity_corpus, os.path.join(save_path, "tokens.pt"))

    def precompute_entity_corpus_embeddings(
            self, 
            save_path: str, 
            tokenized_entity_corpus: Optional[torch.Tensor] = None, 
            batch_size: int = 8, 
            verbose: bool = True,
    ):
        tokenized_entity_corpus = self._tokenize_entity_corpus(verbose=verbose) if tokenized_entity_corpus is None else tokenized_entity_corpus

        # Cast to tensor
        tensor_data_tuple = torch.tensor(tokenized_entity_corpus)
        tensor_data = TensorDataset(tensor_data_tuple)
        sampler = SequentialSampler(tensor_data)
        dataloader = DataLoader(
            tensor_data, sampler=sampler, batch_size=batch_size
        )

        self.corpus_embeddings = torch.zeros(tokenized_entity_corpus.size(0), 1024, dtype=torch.float32)
        for i, batch in enumerate(tqdm(dataloader, desc="Precomputing entity corpus embeddings", disable=not verbose)):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                embeddings = self.model.encode_candidate(*batch).detach().cpu()
                self.corpus_embeddings[i * batch_size: (i + 1) * batch_size] = embeddings
                break

        os.makedirs(save_path, exist_ok=True)
        torch.save(self.corpus_embeddings, os.path.join(save_path, "embeddings.pt"))
        # np.save(os.path.join(save_path, "embeddings.npy"), self.corpus_embeddings)

        self.index = Index(**self.index_config.to_dict())
        self.index.add(np.arange(len(self.corpus_contents)), self.corpus_embeddings)

    def __call__(
            self, 
            texts: List[str]|str = None, 
            passages: List[Passage]|Passage = None,
    ) -> List[Passage]:
        passages = super().__call__(texts=texts, passages=passages)
        pass

    @classmethod
    def from_pretrained(
        cls, 
        model_path: str,
        entity_corpus_path: str,
        precomputed_entity_corpus_path: Optional[str] = None,
        max_candidates: int = 30,
        device: Optional[str] = None,
    ):
        config = ELQConfig(**json.load(open(f"{model_path}/config.json")))
        return cls(
            config=config,
            entity_corpus_path=entity_corpus_path,
            max_candidates=max_candidates,
            precomputed_entity_corpus_path=precomputed_entity_corpus_path,
            device=device,
        )
    

if __name__ == "__main__":
    tokenized_entity_corpus = torch.load("./data/models/entity_linking/elq_wikipedia/elq_entity_corpus/tokens.pt")

    model = ELQ(
        config=ELQConfig(
            bert_model = "bert-large-uncased",
            path_to_model = "./data/models/entity_linking/elq_wikipedia/model.bin",
            max_context_length = 128,
            max_cand_length = 128,
            data_parallel = False,
            no_cuda = False,
        ),
        use_fp16=False,
        entity_corpus_path="./data/entity_corpus/elq_entity_corpus.jsonl",
        # precomputed_entity_corpus_path="./data/models/entity_linking/elq_wikipedia/elq_entity_corpus",
    )
    model.precompute_entity_corpus_embeddings(
        "./data/models/entity_linking/elq_wikipedia/elq_entity_corpus", tokenized_entity_corpus=tokenized_entity_corpus, batch_size=8
    )

    # model = ELQ.from_pretrained(
    #     model_path="./data/models/entity_linking/elq_wikipedia",
    #     entity_corpus_path="./data/entity_corpus/elq_entity_corpus.jsonl",
    #     precomputed_entity_corpus_path="./data/models/entity_linking/elq_wikipedia/elq_entity_corpus",
    #     max_candidates=30,
    #     device="cuda",
    # )