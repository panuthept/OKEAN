import os
import copy
import json
import torch
from tqdm import tqdm
from time import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from okean.utilities.readers import load_entity_corpus
from okean.data_types.basic_types import Passage, Span, Entity
from okean.utilities.general import texts_to_passages, pad_1d_sequence
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from okean.packages.elq_package.biencoder.biencoder import BiEncoderRanker
from okean.modules.entity_linking.baseclass import EntityLinking, EntityLinkingResponse
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
            precomputed_entity_corpus_path: Optional[str] = None,
            device: Optional[str] = None,
            use_fp16: bool = True,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.max_candidates = max_candidates
        self.precomputed_entity_corpus_path = precomputed_entity_corpus_path

        self.model = BiEncoderRanker(config.to_dict())
        self.model.to(self.device)
        self.model.eval()
        if use_fp16: self.model.model.half()
        self.tokenizer = self.model.tokenizer

        self.corpus_contents = load_entity_corpus(entity_corpus_path)
        self.corpus_tokens = None
        self.corpus_embeddings = None

        self._load_precomputed_entity_corpus()

    def _load_precomputed_entity_corpus(self):
        if self.precomputed_entity_corpus_path is None:
            return
        
        precomputed_entity_corpus_tokens_path = os.path.join(self.precomputed_entity_corpus_path, "tokens.pt")
        precomputed_entity_corpus_embeddings_path = os.path.join(self.precomputed_entity_corpus_path, "embeddings.pt")
        if precomputed_entity_corpus_embeddings_path and os.path.exists(precomputed_entity_corpus_embeddings_path):
            print("Loading precomputed entity corpus embeddings...")
            self.corpus_embeddings = torch.load(os.path.join(precomputed_entity_corpus_embeddings_path))
        elif precomputed_entity_corpus_tokens_path and os.path.exists(precomputed_entity_corpus_tokens_path):
            print("Loading precomputed entity corpus tokens...")
            self.corpus_tokens = torch.load(precomputed_entity_corpus_tokens_path)
        print("Done.")

    def _precompute_entity_corpus_tokens(
            self, 
            save_path: str, 
            verbose: bool = True,
            overwrite: bool = False,
    ):
        if os.path.exists(os.path.join(save_path, "tokens.pt")) and not overwrite:
            print("Precomputed entity corpus tokens is already exist. Set `overwrite=True` to recompute.")
            return
        
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

    def precompute_entity_corpus(
            self, 
            save_path: str, 
            batch_size: int = 8, 
            verbose: bool = True,
            overwrite: bool = False,
    ):
        if os.path.exists(os.path.join(save_path, "embeddings.pt")) and not overwrite:
            print("Precomputed entity corpus embeddings is already exist. Set `overwrite=True` to recompute.")
            return
        
        if self.corpus_tokens is None:
            self._precompute_entity_corpus_tokens(save_path=save_path, verbose=verbose, overwrite=overwrite)

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

    def _input_preprocessing(
            self,
            passages: List[Passage],
    ) -> List[torch.Tensor]:
        # Tokenize samples
        max_seq_len = 0
        max_span_num = 0
        encoded_inputs = []
        offset_mappings = []
        mention_bounds = None
        mention_masks = None
        for passage in passages:
            tokenizer_output = self.tokenizer(passage.text, return_offsets_mapping=True)
            tokenizer_output["offset_mapping"] = [[start, end] for start, end in tokenizer_output["offset_mapping"]]

            encoded_input = [101] + tokenizer_output["input_ids"][1:-1][:self.config.max_context_length - 2] + [102]
            offset_mapping = [[0, 0]] + tokenizer_output["offset_mapping"][1:-1][:self.config.max_context_length - 2] + [[0, 0]]
            max_seq_len = max(len(encoded_input), max_seq_len)

            encoded_inputs.append(encoded_input)
            offset_mappings.append(offset_mapping)

            if passage.spans is not None:
                mention_bounds = [] if mention_bounds is None else mention_bounds
                mention_bound = []
                for span in passage.spans:
                    start_token_idx = None
                    end_token_idx = None
                    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                        if token_start <= span.start:
                            start_token_idx = token_idx

                        if token_end <= span.end:
                            end_token_idx = token_idx
                        else:
                            break
                    mention_bound.append([start_token_idx, end_token_idx])
                max_span_num = max(len(mention_bound), max_span_num)
            
                mention_bounds.append(mention_bound)

        # Pad samples
        encoded_inputs = pad_1d_sequence(encoded_inputs, pad_value=0, pad_length=max_seq_len)
        offset_mappings = pad_1d_sequence(offset_mappings, pad_value=[0, 0], pad_length=max_seq_len)
        mention_bounds = pad_1d_sequence(mention_bounds, pad_value=[0, 0], pad_length=max_span_num) if mention_bounds is not None else None

        # Create mention masks
        mention_masks = [[not (start == 0 and end == 0) for start, end in mention_bounds[i]] for i in range(len(mention_bounds))] if mention_bounds is not None else None
        return encoded_inputs, offset_mappings, mention_bounds, mention_masks
    
    def _inference(
            self,
            passages: List[Passage],
            encoded_samples: torch.Tensor,
            offset_mappings: torch.Tensor,
            runtimes: Dict[str, float],
            mention_bounds: Optional[torch.Tensor] = None,
            mention_masks: Optional[torch.Tensor] = None,
            batch_size: int = 8,
            return_candidates: bool = False,
            return_metadata: List[str]|str|bool = False,
    ) -> Tuple[Passage, Any]:
        # Copy input passages
        passages: List[Passage] = copy.deepcopy(passages)
        for passage in passages:
            passage.spans = []

        # Cast to tensor
        tensor_data_tuple = [torch.tensor(encoded_samples), torch.tensor(offset_mappings)]
        if mention_bounds is not None and mention_masks is not None:
            tensor_data_tuple = tensor_data_tuple + [
                torch.tensor(mention_bounds), 
                torch.tensor(mention_masks)
            ]
        tensor_data = TensorDataset(*tensor_data_tuple)
        sampler = SequentialSampler(tensor_data)
        dataloader = DataLoader(
            tensor_data, sampler=sampler, batch_size=batch_size
        )

        metadata_keys = None
        if isinstance(return_metadata, str):
            metadata_keys = [return_metadata]
            return_metadata = True
        elif isinstance(return_metadata, list):
            metadata_keys = return_metadata
            return_metadata = True

        self.corpus_embeddings = self.corpus_embeddings.to(self.device)
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            context_input = batch[0]
            offset_mappings = batch[1]
            mention_bounds = batch[2] if len(batch) > 2 else None
            mention_masks = batch[3] if len(batch) > 2 else None
            with torch.no_grad():
                # Encode input text
                init_time = time()
                embeddings = self.model.encode_context(
                    context_input,
                    gold_mention_bounds=mention_bounds,
                    gold_mention_bounds_mask=mention_masks,
                )
                mention_embeddings = embeddings["mention_reps"]
                mention_masks = embeddings["mention_masks"]         # (batch_size, num_mentions)
                mention_bounds = embeddings["mention_bounds"]       # (batch_size, num_mentions, 2)
                if "mention_logits" in embeddings:
                    mention_logits = embeddings["mention_logits"]   # (batch_size, num_mentions)
                else:
                    mention_logits = torch.full_like(mention_masks, float("inf"), dtype=torch.float32, device=self.device)
                    mention_logits[mention_masks] = torch.tensor(float("-inf"))

                # Get mention embeddings
                mention_embeddings = mention_embeddings[mention_masks]
                runtimes["inference"]["encoding"] += time() - init_time

                # Retrieve candidates
                init_time = time()
                # (num_mentions, max_candidates)
                cand_logits = torch.matmul(mention_embeddings, self.corpus_embeddings.t())
                top_cand_logits_shape, top_cand_indices_shape = cand_logits.topk(self.max_candidates, dim=-1, sorted=True)
                
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
                runtimes["inference"]["search"] += time() - init_time

                # Post-processing
                init_time = time()
                # (batch_size, num_mentions)
                combined_scores = torch.log_softmax(top_cand_logits, -1)[:, :, 0] + torch.sigmoid(mention_logits).log()

                # (num_pred_mentions, )
                pred_mention_masks = (mention_logits > 0).nonzero(as_tuple=True)

                # (num_pred_mentions, )
                pred_mention_logits = mention_logits[pred_mention_masks]
                pred_mention_bounds = mention_bounds[pred_mention_masks]
                pred_combined_scores = combined_scores[pred_mention_masks]
                # (num_pred_mentions, max_candidates)
                pred_cand_logits = top_cand_logits[pred_mention_masks]
                pred_cand_indices = top_cand_indices[pred_mention_masks]

                # (num_pred_mentions, )
                pred_mention_confs = torch.sigmoid(pred_mention_logits)
                # (num_pred_mentions, max_candidates)
                pred_cand_confs = torch.softmax(pred_cand_logits, -1)

                _, sorted_indices = pred_combined_scores.sort(descending=True)

                pred_tokens_mask = torch.zeros_like(context_input)
                for idx in sorted_indices:
                    passage_idx = pred_mention_masks[0][idx]
                    if pred_tokens_mask[passage_idx, pred_mention_bounds[idx][0]:pred_mention_bounds[idx][1]].sum() >= 1:
                        continue

                    span_start = offset_mappings[passage_idx][pred_mention_bounds[idx][0]][0]
                    span_end = offset_mappings[passage_idx][pred_mention_bounds[idx][1]][1]
                    span_text = passages[passage_idx].text[span_start:span_end]

                    passages[passage_idx].spans.append(
                        Span(
                            start=span_start,
                            end=span_end,
                            surface_form=span_text,
                            confident=pred_mention_confs[idx],
                            entity=Entity(
                                identifier=pred_cand_indices[idx][0],
                                confident=pred_cand_confs[idx][0],
                                metadata={k: self.corpus_contents[pred_cand_indices[idx][0]][k] for k in metadata_keys} if return_metadata else None,
                            ),
                            candidates=[
                                Entity(
                                    identifier=pred_cand_indices[idx][cand_idx],
                                    confident=pred_cand_confs[idx][cand_idx],
                                    metadata={k: self.corpus_contents[pred_cand_indices[idx][cand_idx]][k] for k in metadata_keys} if return_metadata else None,
                                )
                            for cand_idx in range(self.max_candidates)] if return_candidates else None,
                        )
                    )
                    pred_tokens_mask[passage_idx, pred_mention_bounds[idx][0]:pred_mention_bounds[idx][1]] = 1
                runtimes["inference"]["post_processing"] += time() - init_time

        # Sort entities by span start
        init_time = time()
        passages = [
            Passage(
                text=passage.text,
                spans=sorted(passage.spans, key=lambda x: x.start),
            )
        for passage in passages]
        runtimes["inference"]["post_processing"] = time() - init_time
        return passages, runtimes

    def __call__(
            self, 
            texts: List[str]|str = None, 
            passages: List[Passage]|Passage = None,
            batch_size: int = 8,
            return_candidates: bool = False,
            return_metadata: List[str]|str|bool = False,
    ) -> EntityLinkingResponse:
        runtimes = {
            "input_preprocessing": 0.0,
            "inference": {
                "encoding": 0.0,
                "search": 0.0,
                "post_processing": 0.0,
            },
        }

        # Prepare input data
        init_time = time()
        passages: List[Passage] = texts_to_passages(texts=texts, passages=passages)
        encoded_inputs, offset_mappings, mention_bounds, mention_masks = self._input_preprocessing(passages=passages)
        runtimes["input_preprocessing"] = time() - init_time

        # Inference
        passages, runtimes = self._inference(
            passages, 
            encoded_inputs, 
            offset_mappings, 
            runtimes=runtimes,
            mention_bounds=mention_bounds,
            mention_masks=mention_masks,
            batch_size=batch_size,
            return_candidates=return_candidates,
            return_metadata=return_metadata
        )
        return EntityLinkingResponse(passages=passages, runtimes=runtimes) 

    @classmethod
    def from_pretrained(
        cls, 
        model_path: str,
        entity_corpus_path: str,
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
        precomputed_entity_corpus_path="./data/models/entity_linking/elq_wikipedia/elq_entity_corpus",
    )
    model.precompute_entity_corpus(save_path="./data/models/entity_linking/elq_wikipedia/elq_entity_corpus")

    # texts = [
    #     "Barack Obama is the former president of the United States.",
    #     "The Eiffel Tower is located in Paris.",
    # ]
    passages = [
        Passage(
            text="Barack Obama is the former president of the United States.", 
            spans=[
                Span(start=0, end=12, surface_form="Barack Obama"),
                Span(start=27, end=57, surface_form="president of the United States"),
            ]
        ),
        Passage(
            text="The Eiffel Tower is located in Paris.",
            spans=[
                Span(start=4, end=16, surface_form="Eiffel Tower"),
            ]
        ),
    ]

    response = model(passages=passages, return_candidates=False, return_metadata=["id"])
    print(response.passages)
    print(response.runtimes)

    # model = ELQ.from_pretrained(
    #     model_path="./data/models/entity_linking/elq_wikipedia",
    #     entity_corpus_path="./data/entity_corpus/elq_entity_corpus.jsonl",
    #     precomputed_entity_corpus_path="./data/models/entity_linking/elq_wikipedia/elq_entity_corpus",
    #     max_candidates=30,
    #     device="cuda",
    # )