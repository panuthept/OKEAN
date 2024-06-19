import torch
import numpy as np
from torch import Tensor
from torch.functional import F
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
from okean.modules.retrieval.baseclass import DenseRetriever, FaissEngineConfig


class mE5(DenseRetriever):
    def __init__(
            self, 
            model_path: str = "intfloat/multilingual-e5-base", 
            search_config: Optional[FaissEngineConfig] = None, 
            corpus_path: Optional[str] = None, 
            device: Optional[str] = None,
    ):
        super().__init__(search_config, corpus_path)
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self.model.to(self.device)

    def _average_pooling(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _encoding(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for i in range(0, len(texts), batch_size):
            inputs = self.tokenizer(texts[i:i + batch_size], max_length=512, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._average_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                embeddings = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()
                all_embeddings[i:i + batch_size] = embeddings
        return embeddings

    def queries_encoding(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        queries = [f"query: {text}" for text in texts]
        return self._encoding(queries, batch_size=batch_size)
    
    def corpus_encoding(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        corpus = [f"passage: {text}" for text in texts]
        return self._encoding(corpus, batch_size=batch_size)
    

if __name__ == "__main__":
    retriever = mE5(corpus_path="./corpus/mE5")
    texts = ["Hello, how are you?", "I am fine, thank you!"]
    retriever.build_corpus(corpus_path="./corpus/mE5", texts=texts)
    queries = ["How are you?", "How are you doing?"]
    results = retriever(queries)
    print(results)