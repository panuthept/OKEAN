# OKEAN - Open Knowledge Enhancement Applications in NLP

## Entity Linking (EL)

```python
from okean.modules.entity_linking.elq import ELQ
from okean.modules.entity_linking.genre import GENRE
from okean.modules.entity_linking.refined import ReFinED

text = "Which member of Black Eyed Peas appeared in Poseidon?"

el_model = ELQ()
doc = el_model(text)
>> Doc(
  text="Which member of Black Eyed Peas appeared in Poseidon?",
  entities=[
    Span(start=16, end=31, text="Black Eyed Peas", entity=Entity(identifier="Q134541")),
    Span(start=44, end=52, text="Poseidon", entity=Entity(identifier="Q906633")),
  ]
)
```

## Entity Disambiguation (ED)
```python
from okean.modules.entity_linking.elq import ELQ
from okean.modules.entity_linking.genre import GENRE
from okean.modules.entity_linking.refined import ReFinED
from okean.modules.entity_disambiguation.blink import BLINK
from okean.modules.entity_disambiguation.global_ed import GlobalED

doc = Doc(
  text="Which member of Black Eyed Peas appeared in Poseidon?",
  entities=[
    Span(start=16, end=31, text="Black Eyed Peas", entity=None),
    Span(start=44, end=52, text="Poseidon", entity=None),
  ]
)

ed_model = BLINK()
doc = ed_model(doc)
>> Doc(
  text="Which member of Black Eyed Peas appeared in Poseidon?",
  entities=[
    Span(start=16, end=31, text="Black Eyed Peas", entity=Entity(identifier="Q134541")),
    Span(start=44, end=52, text="Poseidon", entity=Entity(identifier="Q906633")),
  ]
)
```

## Knowledge Graph Question Answering (KGQA)

#### KAPING (Knowledge-Augmented language model PromptING) 
[Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering (Baek et al., NLRSE 2023)](https://aclanthology.org/2023.nlrse-1.7)

```python
from okean.frameworks.kaping import KAPING
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

answer = kaping_model(text)
>> Fergie
```

## Knowledge-Enhanced Information Retrieval (KEIR)

#### KEIX (Knowledge-Enhanced Information retrieval with query understanding eXpansion) 
```python
from okean.frameworks.keix import KEIX
from okean.modules.retrieval.dpr import DPR
from okean.knowledge_base.wikidata import WikidataKG
from okean.modules.entity_linking.refined import ReFinED

text = "What year did Michael Jordan win his first NBA championship?"

retrieval_model = KEIX(
  el_model=ReFinED(),
  ir_model=DPR(document_corpus_path="<PATH_TO_CORPUS>"),
  kg=WikidataKG()
)

docs = retrieval_model(text)
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
