from pathlib import Path
from dataclasses import dataclass
from okean.modules.entity_linking.refined_package.model_components.refined_model import RefinedModel


@dataclass
class ReFinEDConfig:
    pass


class ReFinED:
    def __init__(
            self, 
            model_path: Path|str,
            model_config: ReFinEDConfig,
    ):
        if isinstance(model_path, str):
            model_path = Path(model_path)

        self.model = RefinedModel.from_pretrained(model_path, model_config)
        

    @classmethod
    def from_pretrained(cls, model_path: Path|str):
        if isinstance(model_path, str):
            model_path = Path(model_path)
        