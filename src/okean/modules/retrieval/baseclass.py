from abc import ABC
from typing import List, Optional
from okean.data_types.basic_types import Doc


class DenseRetriever(ABC):

    def search(self, query: str) -> List[Doc]:
        raise NotImplementedError

    def docs_encoding(self, docs: List[Doc]) -> List[Doc]:
        raise NotImplementedError

    def __call__(
            self, 
            texts: List[str]|str = None, 
            docs: List[Doc]|Doc = None,
    ) -> List[Doc]:
        # Cast `texts` to `docs` if `docs` is not provided
        if docs is None:
            assert texts is not None, "Either `text` or `docs` must be provided."
            if isinstance(texts, list):
                docs = [Doc(text=t) for t in texts]
            else:
                docs = [Doc(text=texts)]
        # Ensure that `docs` is a list of `Doc` objects
        if not isinstance(docs, list):
            docs = [docs]

        docs = self.docs_encoding(docs)
        return docs
    
    def from_pretrained(
        cls, 
        model_path: str,
        document_corpus_path: str,
        device: Optional[str] = None,
    ):
        raise NotImplementedError