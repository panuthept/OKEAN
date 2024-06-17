import os
import torch
from typing import List, Optional
from dataclasses import dataclass
from okean.data_types.basic_types import Doc
from okean.data_types.baseclass import BaseDataType
from okean.modules.entity_linking.baseclass import BaseEntityLinking
from okean.modules.entity_linking.refined_package.model_components.refined_model import RefinedModel
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
            use_precomputed_description_embeddings=config.use_precomputed_descriptions,
            model_description_embeddings_file=config.precomputed_descriptions_emb_file_path,
            max_candidates=config.max_candidates,
        )
        self.model = RefinedModel.from_pretrained(
            model_file=config.model_file_path, 
            model_config_file=config.model_config_file_path,
            preprocessor=self.preprocessor,
            use_precomputed_descriptions=config.use_precomputed_descriptions,
        )
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, doc: List[Doc]|Doc):
        pass
        

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
    model = ReFinED.from_pretrained(model_path="./data/aida_refined")
        