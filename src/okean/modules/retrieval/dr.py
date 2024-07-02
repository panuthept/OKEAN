import os
import copy
import json
import torch
from tqdm import tqdm
from time import time
from torch import Tensor
from torch.functional import F
from dataclasses import dataclass
from typing import List, Dict, Optional
from huggingface_hub import hf_hub_download
from okean.data_types.basic_types import Passage
from okean.utilities.readers import load_corpus_contents
from transformers import AutoTokenizer, AutoModel, AutoConfig
from okean.modules.retrieval.const import DENSE_RETRIEVAL_CONFIGS
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from okean.modules.baseclass import ModuleInterface, ModuleConfig, ModuleResponse


@dataclass
class DenseRetrieverConfig(ModuleConfig):
    max_query_length: int = 512
    max_passage_length: int = 512
    pooling_strategy: str = "average_pooling"
    add_prefix: bool = False


class DenseRetriever(ModuleInterface):
    def __init__(
            self,
            config: DenseRetrieverConfig,
            path_to_models: Optional[Dict[str, str]] = None,
            text_corpus_path: Optional[str] = None, 
            precomputed_text_corpus_path: Optional[str] = None,
            device: Optional[str] = None,
            use_fp16: bool = False,
    ):
        super().__init__(config=config, path_to_models=path_to_models, device=device, use_fp16=use_fp16)
        self.text_corpus_path = text_corpus_path
        self.precomputed_text_corpus_path = precomputed_text_corpus_path
        # Build models
        self._build_models()
        # Load corpus
        self._load_corpus()
    
    def _build_models(self):
        # Build query model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name)
        config = AutoConfig.from_pretrained(self.config.pretrained_model_name)
        self.model = AutoModel.from_config(config)
        self.model.load_state_dict(
            torch.load(self.path_to_models, map_location=self.device), strict=False
        )
        # Move models to device
        self.model.to(self.device)
        # Enable half precision
        if self.use_fp16:
            self.model.half()
        # Set to eval mode
        self.model.eval()

    def _load_corpus(self):
        # Initialize corpus contents, tokens, and embeddings
        self.corpus_contents = None
        self.corpus_tokens = None
        self.corpus_embeddings = None
        # Load corpus contents
        if self.text_corpus_path is not None:
            self.corpus_contents = load_corpus_contents(self.text_corpus_path)
        # Load precomputed corpus tokens and embeddings
        if self.precomputed_text_corpus_path is not None:
            precomputed_text_corpus_tokens_path = os.path.join(self.precomputed_text_corpus_path, "text_corpus_tokens.pt")
            precomputed_text_corpus_embeddings_path = os.path.join(self.precomputed_text_corpus_path, "text_corpus_embeddings.pt")
            if os.path.exists(precomputed_text_corpus_embeddings_path):
                # Load precomputed passage corpus embeddings
                self.corpus_embeddings = torch.load(precomputed_text_corpus_embeddings_path, map_location=self.device)
            elif os.path.exists(precomputed_text_corpus_tokens_path):
                # Load precomputed passage corpus tokens
                self.corpus_tokens = torch.load(precomputed_text_corpus_tokens_path, map_location=self.device)

    def _precopute_corpus_tokens(self) -> Dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            [f"passage: {data['text']}" if self.config.add_prefix else data['text'] for data in self.corpus_contents], 
            max_length=self.config.max_passage_length, 
            padding=True, truncation=True, return_tensors="pt",
        )
        return tokens

    def _precompute_corpus_embeddings(self, batch_size: int = 8, verbose: bool = True) -> Tensor:
        # Create DataLoader
        dataset = TensorDataset(self.corpus_tokens["input_ids"], self.corpus_tokens["attention_mask"])
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)
        # Initialize embeddings
        embeddings = []
        # Compute embeddings
        for input_ids, attention_mask in tqdm(dataloader, desc="Computing corpus embeddings", disable=not verbose):
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            with torch.no_grad():
                last_hidden_state = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
                embeddings.append(self._pooling(last_hidden_state, attention_mask))
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def precompute_corpus(
            self,
            save_path: str,
            batch_size: int = 8, 
            verbose: bool = True,
            overwrite: bool = False,
    ):
        os.makedirs(save_path, exist_ok=True)
        # Check if corpus is already precomputed
        if self.corpus_embeddings is not None:
            if overwrite:
                self.corpus_tokens = None
                self.corpus_embeddings = None
                if verbose:
                    print("Overwriting precomputed corpus.")
            else:
                if verbose:
                    print("Corpus already precomputed. Set `overwrite=True` to overwrite.")
                return
        # Check if corpus tokens are already precomputed
        if self.corpus_tokens is None:
            if verbose:
                print("Precomputing corpus tokens.")
            # Precompute corpus tokens
            self.corpus_tokens = self._precopute_corpus_tokens()
            # Save precomputed corpus tokens
            torch.save(self.corpus_tokens, os.path.join(save_path, "text_corpus_tokens.pt"))
        # Check if corpus embeddings are already precomputed
        if self.corpus_embeddings is None:
            if verbose:
                print("Precomputing corpus embeddings.")
            # Precompute corpus embeddings
            self.corpus_embeddings = self._precompute_corpus_embeddings(batch_size=batch_size, verbose=verbose)
            # Save precomputed corpus embeddings
            torch.save(self.corpus_embeddings, os.path.join(save_path, "text_corpus_embeddings.pt"))

    def _input_preprocessing(
            self,
            passages: List[Passage],
    ) -> Dict[str, torch.Tensor]:
        queries = [f"query: {passage.text}" if self.config.add_prefix else passage.text for passage in passages]
        tokenized_queries = self.tokenizer(queries, max_length=self.config.max_query_length, padding=True, truncation=True, return_tensors="pt")
        return tokenized_queries
    
    def _pooling(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        if self.config.pooling_strategy == "average_pooling":
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            return F.normalize(embeddings, p=2, dim=1)
        else:
            raise ValueError(f"Pooling strategy '{self.config.pooling_strategy}' not supported.")

    def _inference(
            self,
            passages: List[Passage],
            runtimes: Dict[str, float],
            processed_inputs: Dict[str, torch.Tensor],
            batch_size: int = 8,
            top_k: int = 10,
    ) -> ModuleResponse:
        # Create copy of passages
        passages: List[Passage] = copy.deepcopy(passages)
        for passage in passages:
            passage.relevant_passages = []

        # Create DataLoader
        dataset = TensorDataset(processed_inputs["input_ids"], processed_inputs["attention_mask"])
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

        for input_ids, attention_mask in dataloader:
            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
            with torch.no_grad():
                # Encode queries
                init_time = time()
                last_hidden_state = self.model(input_ids, attention_mask=attention_mask).last_hidden_state
                # (queries_num, embedding_dim)
                query_embeddings = self._pooling(last_hidden_state, attention_mask)
                runtimes["inference_encoding"] = time() - init_time

                # Retrieve relevant passages
                init_time = time()
                # (queries_num, passages_num)
                pred_logits = torch.matmul(query_embeddings, self.corpus_embeddings.t())
                pred_scores = F.softmax(pred_logits, dim=-1)
                # (queries_num, top_k)
                top_k_scores, top_k_indices = torch.topk(pred_scores, k=min(top_k, len(self.corpus_contents)), dim=-1, sorted=True)
                runtimes["inference_searching"] = time() - init_time

                # Update passages
                init_time = time()
                for passage_idx, (scores, indices) in enumerate(zip(top_k_scores, top_k_indices)):
                    passages[passage_idx].relevant_passages = [
                        Passage(
                            text=self.corpus_contents[corpus_idx]["text"],
                            confident=score.item(),
                        ) for score, corpus_idx in zip(scores, indices)
                    ]
                runtimes["update_passages"] = time() - init_time
        return ModuleResponse(passages=passages, runtimes=runtimes)

    def save_pretrained(self, path: str):
        # Create directory
        os.makedirs(path, exist_ok=True)
        # Save config
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f)
        # Save models
        torch.save(self.model.state_dict(), os.path.join(path, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        text_corpus_path: Optional[str] = None, 
        precomputed_text_corpus_path: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = False,
    ) -> 'DenseRetriever':
        if os.path.exists(model_name_or_path):
            # Load config
            with open(os.path.join(model_name_or_path, "config.json"), "r") as f:
                config = DenseRetrieverConfig(**json.load(f))
            # Load models
            path_to_models = os.path.join(model_name_or_path, "pytorch_model.bin")
        else:
            if model_name_or_path in DENSE_RETRIEVAL_CONFIGS["E5"]:
                config_dict = DENSE_RETRIEVAL_CONFIGS["E5"]
            else:
                config_dict = DENSE_RETRIEVAL_CONFIGS["Default"]
                print(f"Unknown pretrained model. use default config: {config_dict}")
            # Create config
            config = DenseRetrieverConfig(
                pretrained_model_name=model_name_or_path, **config_dict,
            )
            # Load models
            path_to_models = hf_hub_download(model_name_or_path, "pytorch_model.bin")

        # Return instance
        return cls(
            config=config,
            path_to_models=path_to_models,
            text_corpus_path=text_corpus_path,
            precomputed_text_corpus_path=precomputed_text_corpus_path,
            device=device,
            use_fp16=use_fp16,
        )
    

if __name__ == "__main__":
    model = DenseRetriever.from_pretrained(
        model_name_or_path="sentence-transformers/all-mpnet-base-v2",
        text_corpus_path="./data/text_corpus/test.jsonl",
        precomputed_text_corpus_path="./data/precomputed_text_corpus/test",
    )
    model.precompute_corpus(
        save_path="./data/precomputed_text_corpus/test", batch_size=8, verbose=True, overwrite=True
    )
    
    texts = [
        "That is a happy person",
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