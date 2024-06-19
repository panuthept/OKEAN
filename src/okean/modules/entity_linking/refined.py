import os
import torch
from typing import List, Optional
from okean.utilities.readers import read_entity_corpus
from okean.data_types.basic_types import Passage, Span, Entity
from okean.modules.entity_linking.baseclass import EntityLinking
from okean.packages.refined_package.inference.processor import Refined
from okean.packages.refined_package.data_types.doc_types import Doc as _Doc
from okean.packages.refined_package.data_types.base_types import Span as _Span
from okean.packages.refined_package.doc_preprocessing.preprocessor import PreprocessorInferenceOnly


class ReFinED(EntityLinking):
    def __init__(
            self, 
            model_path: str,
            entity_corpus_path: str,
            max_candidates: int = 30,
            device: Optional[str] = None,
    ):
        model_file_path=os.path.join(model_path, "model.pt")
        model_config_file_path=os.path.join(model_path, "config.json")
        use_precomputed_descriptions=os.path.exists(os.path.join(model_path, "precomputed_entity_descriptions_emb_wikipedia_6269457-300.np"))
        precomputed_descriptions_emb_file_path=os.path.join(model_path, "precomputed_entity_descriptions_emb_wikipedia_6269457-300.np")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.preprocessor = PreprocessorInferenceOnly.from_model_config_file(
            filename=model_config_file_path,
            data_dir="/Users/panuthep/.cache/refined",
            use_precomputed_description_embeddings=use_precomputed_descriptions,
            model_description_embeddings_file=precomputed_descriptions_emb_file_path,
            max_candidates=max_candidates,
        )
        self.refined = Refined(
            model_file_or_model=model_file_path,
            model_config_file_or_model_config=model_config_file_path,
            data_dir="/Users/panuthep/.cache/refined",
            preprocessor=self.preprocessor,
            use_precomputed_descriptions=use_precomputed_descriptions,
            device=device,
        )
        self.entity_corpus = read_entity_corpus(entity_corpus_path)

    def __call__(
            self, 
            texts: List[str]|str = None, 
            passages: List[Passage]|Passage = None,
    ) -> List[Passage]:
        passages = super().__call__(texts=texts, passages=passages)

        _docs: List[_Doc] = self.refined.process_text_batch(
            texts=[d.text for d in passages],
            spanss=[[_Span(text=span.surface_form, start=span.start, ln=span.end - span.start) for span in d.entities] for d in passages] if passages[0].entities is not None else None,
        )

        # Post-process to convert ReFinED object (`_docs`) to standard object (`passages`)
        for _doc, passage in zip(_docs, passages):
            passage.entities = [
                Span(
                    start=_span.start, 
                    end=_span.start + _span.ln,
                    surface_form=_span.text,
                    entities=[
                        Entity(
                            identifier=_entity.wikidata_entity_id if _entity.wikidata_entity_id is not None else "Q0",
                            confident=score,
                            metadata=self.entity_corpus.get(_entity.wikidata_entity_id, None),
                        ) for _entity, score in _span.top_k_predicted_entities
                    ] if _span.top_k_predicted_entities is not None else None
                ) for _span in _doc.spans
            ]
        return passages
    

if __name__ == "__main__":
    from okean.data_types.basic_types import Passage

    el_model = ReFinED(model_path="./data/models/aida_refined", entity_corpus_path="./data/entity_corpus/refined_entity_corpus.jsonl")

    texts = [
        "Michael Jordan published a new paper on machine learning.",
        "Michael Jordan (Michael Irwin Jordan) is a professor at which university?",
        "What year did Michael Jordan win his first NBA championship?",
    ]
    passages = el_model(texts)
    for passage in passages:
        print(passage.text)
        for span in passage.entities:
            print(f"\t{span}")
        print("-" * 100)