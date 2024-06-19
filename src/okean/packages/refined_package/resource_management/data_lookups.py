import pickle
from typing import Mapping, List, Tuple, Dict, Any, Set

import numpy as np
import torch
import ujson as json
from nltk import PunktSentenceTokenizer
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedTokenizer, PreTrainedModel

from okean.packages.refined_package.resource_management.resource_manager import get_mmap_shape
from okean.packages.refined_package.resource_management.lmdb_wrapper import LmdbImmutableDict
from okean.packages.refined_package.resource_management.loaders import load_human_qcode
import os


class LookupsInferenceOnly:

    def __init__(
            self, 
            data_dir: str, 
            use_precomputed_description_embeddings: bool = True,
            transformer_name: str = "roberta_base_model"
    ):
        self.data_dir = data_dir
        self.use_precomputed_description_embeddings = use_precomputed_description_embeddings

        resource_to_file_path = {
            "qcode_idx_to_class_idx": os.path.join(data_dir, "wikipedia_data", "qcode_to_class_tns_6269457-138.np"),
            "descriptions_tns": os.path.join(data_dir, "wikipedia_data", "descriptions_tns.pt"),
            "wiki_pem": os.path.join(data_dir, "wikipedia_data", "pem.lmdb"),
            "class_to_label": os.path.join(data_dir, "wikipedia_data", "class_to_label.json"),
            "human_qcodes": os.path.join(data_dir, "wikipedia_data", "human_qcodes.json"),
            "subclasses": os.path.join(data_dir, "wikipedia_data", "subclasses.lmdb"),
            "qcode_to_idx": os.path.join(data_dir, "wikipedia_data", "qcode_to_idx.lmdb"),
            "class_to_idx": os.path.join(data_dir, "wikipedia_data", "class_to_idx.json"),
            "nltk_sentence_splitter_english": os.path.join(data_dir, "wikipedia_data", "nltk_sentence_splitter_english.pickle"),
            "roberta_base_model": os.path.join(data_dir, "roberta-base", "pytorch_model.bin"),
        }
        self.resource_to_file_path = resource_to_file_path

        # replace all get_file and download_if needed
        # always use resource names that are provided instead of relying on same data_dirs
        # shape = (num_ents, max_num_classes)
        self.qcode_idx_to_class_idx = np.memmap(
            resource_to_file_path["qcode_idx_to_class_idx"],
            shape=get_mmap_shape(resource_to_file_path["qcode_idx_to_class_idx"]),
            mode="r",
            dtype=np.int16,
        )

        if not self.use_precomputed_description_embeddings:
            with open(resource_to_file_path["descriptions_tns"], "rb") as f:
                # (num_ents, desc_len)
                self.descriptions_tns = torch.load(f)
        else:
            # TODO: convert to numpy memmap to save space during training with multiple workers
            self.descriptions_tns = None

        self.pem: Mapping[str, List[Tuple[str, float]]] = LmdbImmutableDict(resource_to_file_path["wiki_pem"])

        with open(resource_to_file_path["class_to_label"], "r") as f:
            self.class_to_label: Dict[str, Any] = json.load(f)

        self.human_qcodes: Set[str] = load_human_qcode(resource_to_file_path["human_qcodes"])

        self.subclasses: Mapping[str, List[str]] = LmdbImmutableDict(resource_to_file_path["subclasses"])

        self.qcode_to_idx: Mapping[str, int] = LmdbImmutableDict(resource_to_file_path["qcode_to_idx"])

        with open(resource_to_file_path["class_to_idx"], "r") as f:
            self.class_to_idx = json.load(f)

        self.index_to_class = {y: x for x, y in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())
        self.max_num_classes_per_ent = self.qcode_idx_to_class_idx.shape[1]
        self.num_classes = len(self.class_to_idx)

        self.qcode_to_wiki = None

        with open(resource_to_file_path["nltk_sentence_splitter_english"], 'rb') as f:
            self.nltk_sentence_splitter_english: PunktSentenceTokenizer = pickle.load(f)

        # can be shared
        self.tokenizers: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            os.path.dirname(resource_to_file_path[transformer_name]),
            add_special_tokens=False,
            add_prefix_space=False,
            use_fast=True,
        )

        self.transformer_model_config = AutoConfig.from_pretrained(
            os.path.dirname(resource_to_file_path[transformer_name])
        )

    def get_transformer_model(self, transformer_name) -> PreTrainedModel:
        # cannot be shared so create a copy
        return AutoModel.from_pretrained(
            os.path.dirname(self.resource_to_file_path[transformer_name])
        )
