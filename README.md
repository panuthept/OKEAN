# OKEAN - Open Knowledge Enhancement Applications in NLP

## Install
```
conda create -n okean python==3.11.4
conda activate okean

# Select the appropriate PyTorch version based on your CUDA version
# CUDA 11.8
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 cpuonly -c pytorch

pip install -e .
```

## Basic Components

## Information Retrieval (IR)

- mE5 ([Multilingual E5 Text Embeddings: A Technical Report. Wang et al., arXiv 2024](https://arxiv.org/pdf/2402.05672))

- BGE-m3 ([M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. Chen et al., arXiv 2024](https://arxiv.org/pdf/2402.03216))

```python
from okean.modules.retrieval import mE5, AVAILABLE_PRETRAINED_IR_MODELS

print(AVAILABLE_PRETRAINED_IR_MODELS)
>> [
  "intfloat/multilingual-e5-small",
  "intfloat/multilingual-e5-base",
  "intfloat/multilingual-e5-large",
  "intfloat/multilingual-e5-large-instruct",
  "BAAI/bge-m3",
  ...
]

ir_model = mE5.from_pretrained(
  model_name_or_path="intfloat/multilingual-e5-base",
)

text = "Which member of Black Eyed Peas appeared in Poseidon?"
response = ir_model(text)
print(response.passages)
>> [
  Passage(
    text="Which member of Black Eyed Peas appeared in Poseidon?",
    relevant_passages: [
      Passage(text="...", confident=0.9),
      Passage(text="...", confident=0.8),
      Passage(text="...", confident=0.7),
    ],
  )
]
```

## Entity Linking (EL)

- ELQ ([Efficient One-Pass End-to-End Entity Linking for Questions. Li et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.522.pdf))
  
- GENRE ([Autoregressive Entity Retrieval. Cao et al., ICLR 2021](https://arxiv.org/pdf/2010.00904))

- ReFinED ([An Efficient Zero-shot-capable Approach to End-to-End Entity Linking. Ayoola et al., NAACL 2022](https://aclanthology.org/2022.naacl-industry.24.pdf))

- KBED ([Improving Entity Disambiguation by Reasoning over a Knowledge Base. Ayoola et al., NAACL 2022](https://aclanthology.org/2022.naacl-main.210.pdf))

```python
from okean.modules.entity_linking import ELQ, AVAILABLE_PRETRAINED_EL_MODELS

print(AVAILABLE_PRETRAINED_EL_MODELS)
>> [
  "panuthept/okean-elq-wikipedia",
  "panuthept/okean-genre-wikipedia",
  "panuthept/okean-genre-aida",
  "panuthept/okean-refined-wikipedia",
  "panuthept/okean-refined-aida",
  "panuthept/okean-kbed-wikipedia"
  "panuthept/okean-kbed-aida"
]

el_model = ELQ.from_pretrained(
  model_name_or_path="panuthept/okean-elq-wikipedia",
)

text = "Which member of Black Eyed Peas appeared in Poseidon?"
response = el_model(text)
print(response.passages)
>> [
  Passage(
    text="Which member of Black Eyed Peas appeared in Poseidon?",
    relevant_entities=[
      Span(
        start=16, end=31, surface_form="Black Eyed Peas", confident=0.5956,
        entity=Entity(identifier=110826, confident=1.0, metadata={'id': {'wikipedia': '210453', 'wikidata': 'Q134541'}})
      ),
      Span(
        start=44, end=53, surface_form="Poseidon?", confident=0.2635,
        entity=Entity(identifier=664979, confident=0.9639, metadata={'id': {'wikipedia': '2688309', 'wikidata': 'Q906633'}})
      ),
    ]
  )
]
```

## Knowledge-Enhanced Information Retrieval (KEIR)

[Enhancing Late Interaction with Informative Entities for Passage Retrieval (Fang et al., ECIR 2024)](https://keirworkshop.github.io/assets/files/keir_4.pdf)

```python
from okean.modules.keir.colluke import ColLUKE
from okean.modules.entity_linking.elq import ELQ
from okean.modules.retrieval.colbert import ColBERT
from okean.knowledge_base.wikidata import WikidataKG

model = ColLUKE(
  el_model=ELQ(),
  ir_model=ColBERT(),
  kg=WikidataKG()
)

text = "Which member of Black Eyed Peas appeared in Poseidon?"
response = model(text)
```

## Knowledge-Base Question Answering (KBQA)

[Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering (Baek et al., NLRSE 2023)](https://aclanthology.org/2023.nlrse-1.7)

```python
from okean.modules.kgqa.kaping import KAPING
from okean.modules.ranking.mpnet import MPNet
from okean.modules.generative_llms.t5 import T5
from okean.knowledge_base.wikidata import WikidataKG
from okean.modules.entity_linking.refined import ReFinED

kaping_model = KAPING(
  el_model=ReFinED(),
  ranking_model=MPNet(),
  llm_model=T5(),
  kg=WikidataKG()
)

text = "Which member of Black Eyed Peas appeared in Poseidon?"
response = kaping_model(text)
print(response.answer)
>> Fergie
```

## Training Toolkit
```python
from okean.modules.entity_linking.refined import ReFinED
from okean.datasets.entity_linking import EntityLinkingDataset
from okean.trainers.entity_linking.refined import ReFinEDTrainer, ReFinEDTrainerConfig

train_set = EntityLinkingDataset("<PATH_TO_DATASET>")
dev_set = EntityLinkingDataset("<PATH_TO_DATASET>")

config = ReFinEDTrainerConfig()
trainer = ReFinEDTrainer(config)
trainer.train(train_set=train_set, dev_set=dev_set)

trainer.save_pretrained("<PATH_TO_SAVE>")
```

## Evaluation Toolkit
```python
from okean.modules.entity_linking.refined import ReFinED
from okean.datasets.entity_linking import EntityLinkingDataset
from okean.benchmarks.entity_linking import EntityLinkingBenchmark

train_set = EntityLinkingDataset("<PATH_TO_DATASET>") 
test_sets = {
  "AIDA": EntityLinkingDataset("<PATH_TO_DATASET>"),
  "Tweeki": EntityLinkingDataset("<PATH_TO_DATASET>"),
  "Mintaka": EntityLinkingDataset("<PATH_TO_DATASET>"),
}

el_model = ReFinED(model_path"<PATH_TO_MODEL>")
el_benchmark = EntityLinkingBenchmark(
  metrics=["precision", "recall", "f1"],
  entity_corpus_path="<PATH_TO_CORPUS>",
  train_entity_prior=train_set.statistics("entity_prior"),
  report_overshadowing=True,
  report_runtime=True,
  inkb_only=True,
  verbose=True,
)

results = el_benchmark(el_model, test_sets)
>> {
  "AIDA": {
    "precision": {"shadow": 0.6, "top": 0.9, "all": 0.7},
    "recall": {"shadow": 0.6, "top": 0.9, "all": 0.7},
    "f1": {"shadow": 0.6, "top": 0.9, "all": 0.7},
    "shadow_prop": 0.1,
    "runtime": 0.5,
  },
  "Tweeki": {
    "precision": {"shadow": 0.6, "top": 0.9, "all": 0.7},
    "recall": {"shadow": 0.6, "top": 0.9, "all": 0.7},
    "f1": {"shadow": 0.6, "top": 0.9, "all": 0.7},
    "shadow_prop": 0.2,
    "runtime": 0.6,
  },
  "Mintaka": {
    "precision": {"shadow": 0.6, "top": 0.9, "all": 0.7},
    "recall": {"shadow": 0.6, "top": 0.9, "all": 0.7},
    "f1": {"shadow": 0.6, "top": 0.9, "all": 0.7},
    "shadow_prop": 0.3,
    "runtime": 0.7,
  }
}
```
