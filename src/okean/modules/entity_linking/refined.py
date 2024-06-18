import os
import torch
from typing import List, Optional
from dataclasses import dataclass
from okean.data_types.baseclass import BaseDataType
from okean.utilities.readers import read_entity_corpus
from okean.data_types.basic_types import Doc, Span, Entity
from okean.modules.entity_linking.baseclass import BaseEntityLinking
from okean.modules.entity_linking.refined_package.inference.processor import Refined
from okean.modules.entity_linking.refined_package.data_types.doc_types import Doc as _Doc
from okean.modules.entity_linking.refined_package.data_types.base_types import Span as _Span
from okean.modules.entity_linking.refined_package.doc_preprocessing.preprocessor import PreprocessorInferenceOnly


@dataclass
class ReFinEDConfig(BaseDataType):
    model_file_path: str
    model_config_file_path: str
    use_precomputed_descriptions: bool = False
    precomputed_descriptions_emb_file_path: Optional[str] = None
    max_candidates: int = 30


class ReFinED(BaseEntityLinking):
    """
    This class is a wrapper for the ReFinED model.
    """
    def __init__(
            self, 
            config: ReFinEDConfig,
            entity_corpus_path: str,
            device: Optional[str] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.preprocessor = PreprocessorInferenceOnly.from_model_config_file(
            filename=config.model_config_file_path,
            data_dir="/Users/panuthep/.cache/refined",
            use_precomputed_description_embeddings=config.use_precomputed_descriptions,
            model_description_embeddings_file=config.precomputed_descriptions_emb_file_path,
            max_candidates=config.max_candidates,
        )
        self.refined = Refined(
            model_file_or_model=config.model_file_path,
            model_config_file_or_model_config=config.model_config_file_path,
            data_dir="/Users/panuthep/.cache/refined",
            preprocessor=self.preprocessor,
            use_precomputed_descriptions=config.use_precomputed_descriptions,
            device=device,
        )
        self.entity_corpus = read_entity_corpus(entity_corpus_path)

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

        _docs: List[_Doc] = self.refined.process_text_batch(
            texts=[d.text for d in docs],
            spanss=[[_Span(text=span.surface_form, start=span.start, ln=span.end - span.start) for span in d.entities] for d in docs] if docs[0].entities is not None else None,
        )

        # Post-process to convert ReFinED object (`_docs`) to standard object (`docs`)
        for _doc, doc in zip(_docs, docs):
            doc.entities = [
                Span(
                    start=_span.start, 
                    end=_span.start + _span.ln,
                    surface_form=_span.text,
                    entity=Entity(
                        identifier=_span.top_k_predicted_entities[0][0].wikidata_entity_id if _span.top_k_predicted_entities[0][0].wikidata_entity_id is not None else "Q0",
                        confident=_span.top_k_predicted_entities[0][1],
                        metadata=self.entity_corpus.get(_span.top_k_predicted_entities[0][0].wikidata_entity_id, None),
                    ),
                    candidates=[
                        Entity(
                            identifier=_entity.wikidata_entity_id if _entity.wikidata_entity_id is not None else "Q0",
                            confident=score,
                            metadata=self.entity_corpus.get(_entity.wikidata_entity_id, None),
                        ) for _entity, score in _span.top_k_predicted_entities
                    ] if _span.top_k_predicted_entities is not None else None
                ) for _span in _doc.spans
            ]
        return docs
        

    @classmethod
    def from_pretrained(
        cls, 
        model_path: str,
        entity_corpus_path: str,
        device: Optional[str] = None,
    ):
        config = ReFinEDConfig(
            model_file_path=os.path.join(model_path, "model.pt"),
            model_config_file_path=os.path.join(model_path, "config.json"),
            use_precomputed_descriptions=os.path.exists(os.path.join(model_path, "precomputed_entity_descriptions_emb_wikipedia_6269457-300.np")),
            precomputed_descriptions_emb_file_path=os.path.join(model_path, "precomputed_entity_descriptions_emb_wikipedia_6269457-300.np"),
        )
        return cls(config, entity_corpus_path, device=device)
    

if __name__ == "__main__":
    from okean.data_types.basic_types import Doc

    el_model = ReFinED.from_pretrained(model_path="./data/models/aida_refined", entity_corpus_path="./data/entity_corpus/refined_entity_corpus.jsonl")

    texts = [
        "Michael Jordan published a new paper on machine learning.",
        "Michael Jordan (Michael Irwin Jordan) is a professor at which university?",
        "What year did Michael Jordan win his first NBA championship?",
    ]
    docs = el_model(texts)
    for doc in docs:
        print(doc.text)
        for span in doc.entities:
            span.candidates = None
            print(f"\t{span}")
        print("-" * 100)