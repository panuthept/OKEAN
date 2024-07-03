import torch
from typing import List, Dict
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from okean.modules.retrieval.baseclass import Retrieval, RetrievalConfig



@dataclass
class E5Config(RetrievalConfig):
    pass


class E5(Retrieval):
    @property
    def available_models(self):
        return [
            "intfloat/multilingual-e5-small",
            "intfloat/multilingual-e5-base",
            "intfloat/multilingual-e5-large",
            "intfloat/e5-small",
            "intfloat/e5-base",
            "intfloat/e5-large",
            "intfloat/e5-small-v2",
            "intfloat/e5-base-v2",
            "intfloat/e5-large-v2",
        ]

    def _tokenize_corpus(self, corpus: List[str]) -> Dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            [f"passage: {text}" for text in corpus], 
            max_length=self.config.max_passage_length, padding=True, truncation=True, return_tensors="pt",
        )
        return tokens
    
    def _tokenize_queries(self, queries: List[str]) -> Dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            [f"query: {text}" for text in queries], 
            max_length=self.config.max_query_length, padding=True, truncation=True, return_tensors="pt",
        )
        return tokens
    
    @staticmethod
    def _load_from_hub(hub_path: str):
        # Load config
        config = E5Config(
            pretrained_model_name=hub_path,
        )
        # Load model
        path_to_models = hf_hub_download(hub_path, "pytorch_model.bin")
        return config, path_to_models
    

if __name__ == "__main__":
    model = E5.from_pretrained(
        model_name_or_path="intfloat/multilingual-e5-base",
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