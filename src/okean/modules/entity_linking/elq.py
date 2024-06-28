import os
import copy
import json
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from okean.data_types.config_types import IndexConfig
from okean.utilities.readers import load_entity_corpus
from usearch.index import Index, BatchMatches, Matches, search
from okean.data_types.basic_types import Passage, Span, Entity
from okean.modules.entity_linking.baseclass import EntityLinking
from okean.utilities.general import texts_to_passages, pad_1d_sequence
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from okean.packages.elq_package.biencoder.biencoder import BiEncoderRanker
from okean.packages.elq_package.biencoder.data_process import get_candidate_representation


@dataclass
class ELQConfig:
    lowercase: bool = True
    bert_model: str = "bert-large-uncased"
    path_to_model: Optional[str] = None
    load_cand_enc_only: bool = False
    mention_aggregation_type: str = "all_avg"
    mention_scoring_method: str = "qa_linear"
    max_context_length: int = 128
    max_cand_length: int = 128
    max_mention_length: int = 32
    out_dim: int = 1
    pull_from_layer: int = -1
    add_linear: bool = False
    freeze_cand_enc: bool = True, 
    data_parallel: bool = False
    no_cuda: bool = False

    def to_dict(self):
        return self.__dict__


class ELQ(EntityLinking):
    def __init__(
            self, 
            config: ELQConfig,
            entity_corpus_path: str,
            max_candidates: int = 30,
            index_config: Optional[IndexConfig] = None, 
            precomputed_entity_corpus_index_path: Optional[str] = None,
            precomputed_entity_corpus_tokens_path: Optional[str] = None,
            precomputed_entity_corpus_embeddings_path: Optional[str] = None,
            device: Optional[str] = None,
            use_fp16: bool = True,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.max_candidates = max_candidates
        self.precomputed_entity_corpus_index_path = precomputed_entity_corpus_index_path
        self.precomputed_entity_corpus_tokens_path = precomputed_entity_corpus_tokens_path
        self.precomputed_entity_corpus_embeddings_path = precomputed_entity_corpus_embeddings_path

        self.model = BiEncoderRanker(config.to_dict())
        self.model.to(self.device)
        self.model.eval()
        if use_fp16: self.model.model.half()
        self.tokenizer = self.model.tokenizer

        if index_config is None: index_config = IndexConfig(ndim=1024, metric="ip", dtype="f32")
        self.index_config = index_config

        self.corpus_contents = load_entity_corpus(entity_corpus_path)
        self.corpus_tokens = None
        self.corpus_embeddings = None
        self.index = None

        if self.precomputed_entity_corpus_index_path and os.path.exists(self.precomputed_entity_corpus_index_path):
            print("Loading precomputed entity corpus index...")
            self.index = Index.restore(self.precomputed_entity_corpus_index_path)
        elif self.precomputed_entity_corpus_embeddings_path and os.path.exists(self.precomputed_entity_corpus_embeddings_path):
            print("Loading precomputed entity corpus embeddings...")
            self.corpus_embeddings = torch.load(os.path.join(self.precomputed_entity_corpus_embeddings_path))
        elif self.precomputed_entity_corpus_tokens_path and os.path.exists(self.precomputed_entity_corpus_tokens_path):
            print("Loading precomputed entity corpus tokens...")
            self.corpus_tokens = torch.load(self.precomputed_entity_corpus_tokens_path)
        print("Done initializing ELQ.")
    
    def precompute_entity_corpus_tokens(
            self, 
            save_path: str, 
            verbose: bool = True
    ):
        titles = [entity["name"] for entity in self.corpus_contents]
        descs = [entity["desc"] for entity in self.corpus_contents]

        self.corpus_tokens = []
        for title, desc in tqdm(zip(titles, descs), total=len(titles), desc="Tokenizing entities", disable=not verbose):
            tokenized_entity = get_candidate_representation(
                desc, self.tokenizer, self.config.max_cand_length, title
            )["ids"][0]
            self.corpus_tokens.append(tokenized_entity)

        os.makedirs(save_path, exist_ok=True)
        torch.save(self.corpus_tokens, os.path.join(save_path, "tokens.pt"))

    def precompute_entity_corpus_embeddings(
            self, 
            save_path: str, 
            batch_size: int = 8, 
            verbose: bool = True,
    ):
        if self.corpus_tokens is None:
            self.precompute_entity_corpus_tokens(save_path=save_path, verbose=verbose)

        # Cast to tensor
        tensor_data_tuple = torch.tensor(self.corpus_tokens)
        tensor_data = TensorDataset(tensor_data_tuple)
        sampler = SequentialSampler(tensor_data)
        dataloader = DataLoader(
            tensor_data, sampler=sampler, batch_size=batch_size
        )

        self.corpus_embeddings = torch.zeros(self.corpus_tokens.size(0), 1024, dtype=torch.float32)
        for i, batch in enumerate(tqdm(dataloader, desc="Precomputing entity corpus embeddings", disable=not verbose)):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                embeddings = self.model.encode_candidate(*batch).detach().cpu()
                self.corpus_embeddings[i * batch_size: (i + 1) * batch_size] = embeddings

        os.makedirs(save_path, exist_ok=True)
        torch.save(self.corpus_embeddings, os.path.join(save_path, "embeddings.pt"))

    def precompute_entity_corpus_index(
            self, 
            save_path: str,
            verbose: bool = True,
    ):
        if self.corpus_embeddings is None:
            self.precompute_entity_corpus_embeddings(save_path=save_path, verbose=verbose)

        self.index = Index(**self.index_config.to_dict())
        self.index.add(np.arange(len(self.corpus_contents)), self.corpus_embeddings.numpy(), log="Creating index" if verbose else False)
        self.index.save(os.path.join(save_path, "index.usearch"))

    def _input_preprocessing(
            self,
            passages: List[Passage],
            batch_size: int = 8,
    ) -> DataLoader:
        # Tokenize samples
        max_seq_len = 0
        encoded_samples = []
        offset_mappings = []
        for passage in passages:
            tokenizer_output = self.tokenizer(passage.text, return_offsets_mapping=True)
            encoded_sample = [101] + tokenizer_output["input_ids"][1:-1][:self.config.max_context_length - 2] + [102]
            offset_mapping = [(0, 0)] + tokenizer_output["offset_mapping"][1:-1][:self.config.max_context_length - 2] + [(0, 0)]
            max_seq_len = max(len(encoded_sample), max_seq_len)
            encoded_samples.append(encoded_sample)
            offset_mappings.append(offset_mapping)
        print(f"encoded_samples:\n{encoded_samples}\n{len(encoded_samples)}")
        print(f"offset_mappings:\n{offset_mappings}\n{len(offset_mappings)}")

        # Pad samples
        padded_encoded_samples = pad_1d_sequence(encoded_samples, pad_value=0, pad_length=max_seq_len)
        padded_offset_mappings = pad_1d_sequence(offset_mappings, pad_value=(0, 0), pad_length=max_seq_len)
        print(f"padded_encoded_samples:\n{padded_encoded_samples}\n{len(padded_encoded_samples)}")
        print(f"padded_offset_mappings:\n{padded_offset_mappings}\n{len(padded_offset_mappings)}")

        # Cast to tensor
        tensor_data_tuple = torch.tensor([padded_encoded_samples, padded_offset_mappings])
        tensor_data = TensorDataset(*tensor_data_tuple)
        sampler = SequentialSampler(tensor_data)
        dataloader = DataLoader(
            tensor_data, sampler=sampler, batch_size=batch_size
        )
        return dataloader

    def __call__(
            self, 
            texts: List[str]|str = None, 
            passages: List[Passage]|Passage = None,
            batch_size: int = 8,
    ) -> List[Passage]:
        passages: List[Passage] = texts_to_passages(texts=texts, passages=passages)
        output_passages = copy.deepcopy(passages)
        for passage in output_passages:
            passage.entities = []
        
        # Prepare input data
        dataloader = self._input_preprocessing(passages=passages, batch_size=batch_size)

        # Inference
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            context_input = batch[0]
            offset_mappings = batch[1]
            with torch.no_grad():
                # Encode input text
                # print(context_input)
                embeddings = self.model.encode_context(context_input)
                mention_embeddings = embeddings["mention_reps"]
                mention_masks = embeddings["mention_masks"]
                mention_logits = embeddings["mention_logits"]   # (batch_size, num_mentions)
                mention_bounds = embeddings["mention_bounds"]   # (batch_size, num_mentions, 2)

                # Get mention embeddings
                mention_embeddings = mention_embeddings[mention_masks]
                # print(f"mention_embeddings:\n{mention_embeddings}\n{mention_embeddings.size()}")
                # print(f"mention_masks:\n{mention_masks}\n{mention_masks.size()}")
                # print(f"mention_logits:\n{mention_logits}\n{mention_logits.size()}")
                # print(f"mention_bounds:\n{mention_bounds}\n{mention_bounds.size()}")

                # Retrieve candidates
                if self.index is None:
                    cand_logits, _, _ = self.model.score_candidate(
                        context_input, None,
                        text_encs=mention_embeddings,
                        cand_encs=self.corpus_embeddings.to(self.device),
                    )
                    top_cand_logits_shape, top_cand_indices_shape = cand_logits.topk(self.max_candidates, dim=-1, sorted=True)
                    # matches: Union[BatchMatches, Matches] = search(
                    #     self.corpus_embeddings.numpy(), mention_embeddings.numpy(), count=self.max_candidates, metric="ip", exact=True
                    # )
                else:
                    matches: Union[BatchMatches, Matches] = self.index.search(mention_embeddings.numpy(), count=self.max_candidates)
                    if isinstance(matches, Matches):
                        matches = [matches]

                # print(f"top_cand_logits_shape:\n{top_cand_logits_shape}\n{top_cand_logits_shape.size()}")
                # print(f"top_cand_indices_shape:\n{top_cand_indices_shape}\n{top_cand_indices_shape.size()}")

                # (batch_size, num_mentions, max_candidates)
                top_cand_logits = torch.zeros(
                    mention_logits.size(0), mention_logits.size(1), self.max_candidates
                ).to(self.device, top_cand_logits_shape.dtype)
                top_cand_logits[mention_masks] = top_cand_logits_shape

                # (batch_size, num_mentions, max_candidates)
                top_cand_indices = torch.zeros(
                    mention_logits.size(0), mention_logits.size(1), self.max_candidates
                ).to(self.device, top_cand_indices_shape.dtype)
                top_cand_indices[mention_masks] = top_cand_indices_shape

                # (batch_size, num_mentions)
                combined_scores = torch.log_softmax(top_cand_logits, -1)[:, :, 0] + torch.sigmoid(mention_logits).log()
                # print(f"combined_scores:\n{combined_scores}\n{combined_scores.size()}")

                pred_mention_masks = (mention_logits > 0).nonzero(as_tuple=True)
                # print(f"pred_mention_masks:\n{pred_mention_masks}")
                # (batch_size, num_pred_mentions)
                pred_mention_bounds = mention_bounds[pred_mention_masks]
                pred_combined_scores = combined_scores[pred_mention_masks]
                # (batch_size, num_pred_mentions, max_candidates)
                pred_cand_logits = top_cand_logits[pred_mention_masks]
                pred_cand_indices = top_cand_indices[pred_mention_masks]
                # print(f"pred_mention_bounds:\n{pred_mention_bounds}\n{pred_mention_bounds.size()}")
                # print(f"pred_combined_scores:\n{pred_combined_scores}\n{pred_combined_scores.size()}")
                # print(f"pred_cand_logits:\n{pred_cand_logits}\n{pred_cand_logits.size()}")
                # print(f"pred_cand_indices:\n{pred_cand_indices}\n{pred_cand_indices.size()}")

                _, sorted_indices = pred_combined_scores.sort(descending=True)

                # final_cand_logits = []
                # final_cand_indices = []
                # final_mention_bounds = []
                pred_tokens_mask = torch.zeros_like(context_input)
                # Remove special tokens [CLS] and [SEP]
                context_input = context_input[:, 1:-1]
                # print(f"sorted_indices:\n{sorted_indices}\n{sorted_indices.size()}")
                for idx in sorted_indices:
                    passage_idx = pred_mention_masks[0][idx]
                    if pred_tokens_mask[passage_idx, pred_mention_bounds[idx][0]:pred_mention_bounds[idx][1]].sum() >= 1:
                        continue
                    # final_cand_logits.append(pred_cand_logits[idx])
                    # final_cand_indices.append(pred_cand_indices[idx])
                    # final_mention_bounds.append(pred_mention_bounds[idx])
                    # span_start =  # Map to original character index
                    output_passages[passage_idx].entities.append(
                        Span(
                            start=offset_mappings[passage_idx][pred_mention_bounds[idx][0]][0],
                            end=offset_mappings[passage_idx][pred_mention_bounds[idx][1]][1],
                            surface_form=self.tokenizer.decode(context_input[passage_idx, pred_mention_bounds[idx][0]:pred_mention_bounds[idx][1]]),
                            entities=[
                                Entity(
                                    identifier=pred_cand_indices[idx][cand_idx],
                                    confident=pred_cand_logits[idx][cand_idx],
                                    # metadata=self.corpus_contents[pred_cand_indices[idx][cand_idx]],
                                )
                            for cand_idx in range(self.max_candidates)]
                        )
                    )
                    pred_tokens_mask[passage_idx, pred_mention_bounds[idx][0]:pred_mention_bounds[idx][1]] = 1
                print(f"output_passages:\n{output_passages}")
                # final_cand_logits = torch.stack(final_cand_logits)
                # final_cand_indices = torch.stack(final_cand_indices)
                # final_mention_bounds = torch.stack(final_mention_bounds)
                # print(f"final_cand_logits:\n{final_cand_logits}\n{final_cand_logits.size()}")
                # print(f"final_cand_indices:\n{final_cand_indices}\n{final_cand_indices.size()}")
                # print(f"final_mention_bounds:\n{final_mention_bounds}\n{final_mention_bounds.size()}")

                # Remove special tokens [CLS] and [SEP]
                # context_input = context_input[:, 1:-1]
                


    @classmethod
    def from_pretrained(
        cls, 
        model_path: str,
        entity_corpus_path: str,
        index_config: Optional[IndexConfig] = None, 
        precomputed_entity_corpus_index_path: Optional[str] = None,
        precomputed_entity_corpus_tokens_path: Optional[str] = None,
        precomputed_entity_corpus_embeddings_path: Optional[str] = None,
        max_candidates: int = 30,
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        config = ELQConfig(**json.load(open(f"{model_path}/config.json")))
        return cls(
            config=config,
            entity_corpus_path=entity_corpus_path,
            max_candidates=max_candidates,
            index_config=index_config,
            precomputed_entity_corpus_index_path=precomputed_entity_corpus_index_path,
            precomputed_entity_corpus_tokens_path=precomputed_entity_corpus_tokens_path,
            precomputed_entity_corpus_embeddings_path=precomputed_entity_corpus_embeddings_path,
            device=device,
            use_fp16=use_fp16,
        )
    

if __name__ == "__main__":
    model = ELQ(
        config=ELQConfig(
            bert_model = "bert-large-uncased",
            path_to_model = "./data/models/entity_linking/elq_wikipedia/model.bin",
            max_context_length = 128,
            max_cand_length = 128,
            data_parallel = False,
            no_cuda = False,
        ),
        use_fp16=False,
        entity_corpus_path="./data/entity_corpus/elq_entity_corpus.jsonl",
        precomputed_entity_corpus_tokens_path="./data/models/entity_linking/elq_wikipedia/elq_entity_corpus/tokens.pt",
        precomputed_entity_corpus_embeddings_path="./data/models/entity_linking/elq_wikipedia/elq_entity_corpus/embeddings.pt",
    )
    # model.precompute_entity_corpus_index(save_path="./data/models/entity_linking/elq_wikipedia/elq_entity_corpus")

    texts = [
        "Barack Obama is the former president of the United States.",
        "The Eiffel Tower is located in Paris.",
    ]
    model(texts=texts)
    # model = ELQ.from_pretrained(
    #     model_path="./data/models/entity_linking/elq_wikipedia",
    #     entity_corpus_path="./data/entity_corpus/elq_entity_corpus.jsonl",
    #     precomputed_entity_corpus_path="./data/models/entity_linking/elq_wikipedia/elq_entity_corpus",
    #     max_candidates=30,
    #     device="cuda",
    # )