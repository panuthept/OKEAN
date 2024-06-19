from transformers import AutoTokenizer, MPNetModel
from okean.modules.retrieval.baseclass import DenseRetriever


class MPNet(DenseRetriever):
    def __init__(self, model_path: str = "microsoft/mpnet-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = MPNetModel.from_pretrained(model_path)

    @classmethod
    def from_pretrained(cls, model_path: str, document_corpus_path: str, device: str | None = None):
        return super().from_pretrained(model_path, document_corpus_path, device)