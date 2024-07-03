import torch
from time import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from okean.data_types.basic_types import Passage
from okean.utilities.general import texts_to_passages


@dataclass
class ModuleConfig:
    pretrained_model_name: str

    def to_dict(self):
        return self.__dict__


@dataclass
class ModuleResponse:
    passages: List[Passage]
    runtimes: Dict[str, float]


class ModuleInterface(ABC):
    @abstractmethod
    def __init__(
            self, 
            config: ModuleConfig,
            path_to_models: Optional[Dict[str, str]] = None,
            device: Optional[str] = None,
            use_fp16: bool = False,
    ):
        self.config = config
        self.path_to_models = path_to_models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.use_fp16 = use_fp16

    @abstractmethod
    def _build_models(self):
        raise NotImplementedError

    @abstractmethod
    def _input_preprocessing(
            self,
            passages: List[Passage],
            batch_size: int = 8,
            **kwargs,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _inference(
            self,
            passages: List[Passage],
            runtimes: Dict[str, float],
            processed_inputs: Any,
            batch_size: int = 8,
            **kwargs,
    ) -> ModuleResponse:
        raise NotImplementedError

    def __call__(
            self, 
            texts: List[str] = None, 
            passages: List[Passage] = None, 
            batch_size: int = 8,
            **kwargs,
    ) -> ModuleResponse:
        runtimes = {
            "input_preprocessing": 0.0,
            "inference": 0.0,
        }

        # Prepare input data
        init_time = time()
        passages: List[Passage] = texts_to_passages(texts=texts, passages=passages)
        processed_inputs = self._input_preprocessing(
            passages=passages, 
            batch_size=batch_size,
            **kwargs
        )
        runtimes["input_preprocessing"] = time() - init_time

        # Inference
        init_time = time()
        response: ModuleResponse = self._inference(
            passages, 
            runtimes,
            processed_inputs,
            batch_size=batch_size,
            **kwargs,
        )
        runtimes["inference"] = time() - init_time
        return response
    
    @abstractmethod
    def save_pretrained(self, path: str):
        raise NotImplementedError
    
    @classmethod
    def from_pretrained(
        cls, 
        model_name_or_path: str,
        device: Optional[str] = None,
        use_fp16: bool = True,
    ) -> 'ModuleInterface':
        raise NotImplementedError