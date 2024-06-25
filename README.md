# OKEAN - Open Knowledge Enhancement Applications in NLP

<img width="879" alt="okean" src="https://github.com/panuthept/OKEAN/assets/28400944/62ddb72d-e977-469c-8fde-380227724213">


## Basic Components

### Information Retrieval (IR)

```python
from okean.modules.retrieval.elq import mE5

corpus = [
  ...
]
text = "Which member of Black Eyed Peas appeared in Poseidon?"

ir_model = mE5()
ir_model.build_corpus(corpus_path="<PATH_TO_CORPUS>", texts=corpus)
response = ir_model(text)
>> Passage(
  text="Which member of Black Eyed Peas appeared in Poseidon?",
  relevant_passages=[
    Passage(text="...", confident=0.9),
    Passage(text="...", confident=0.8),
    Passage(text="...", confident=0.7),
  ]
)
```

### Entity Linking (EL)

```python
from okean.modules.entity_linking.elq import ELQ
from okean.modules.entity_linking.genre import GENRE
from okean.modules.entity_linking.refined import ReFinED

text = "Which member of Black Eyed Peas appeared in Poseidon?"

el_model = ELQ()
response = el_model(text)
>> Passage(
  text="Which member of Black Eyed Peas appeared in Poseidon?",
  entities=[
    Span(start=16, end=31, text="Black Eyed Peas", entity=Entity(identifier="Q134541")),
    Span(start=44, end=52, text="Poseidon", entity=Entity(identifier="Q906633")),
  ]
)
```

## Knowledge-Enhanced Information Retrieval (KEIR)

[ColLUKE - Enhancing Late Interaction with Informative Entities for Passage Retrieval (Fang et al., ECIR 2024)](https://keirworkshop.github.io/assets/files/keir_4.pdf)

```python
from okean.modules.keir.colluke import ColLUKE
from okean.modules.entity_linking.elq import ELQ
from okean.modules.retrieval.colbert import ColBERT
from okean.knowledge_base.wikidata import WikidataKG

text = "Which member of Black Eyed Peas appeared in Poseidon?"

model = ColLUKE(
  el_model=ELQ(),
  ir_model=ColBERT(),
  kg=WikidataKG()
)

response = model(text)
```

## Knowledge-Base Question Answering (KBQA)

[KAPING - Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering (Baek et al., NLRSE 2023)](https://aclanthology.org/2023.nlrse-1.7)

```python
from okean.modules.kgqa.kaping import KAPING
from okean.modules.ranking.mpnet import MPNet
from okean.modules.generative_llms.t5 import T5
from okean.knowledge_base.wikidata import WikidataKG
from okean.modules.entity_linking.refined import ReFinED

text = "Which member of Black Eyed Peas appeared in Poseidon?"

kaping_model = KAPING(
  el_model=ReFinED(),
  ranking_model=MPNet(),
  llm_model=T5(),
  kg=WikidataKG()
)

response = kaping_model(text)
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
