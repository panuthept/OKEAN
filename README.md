# OKEAN - Open Knowledge Enhancement Applications in NLP

## Knowledge Graph Question Answering (KGQA)
#### [Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering (Baek et al., NLRSE 2023)](https://aclanthology.org/2023.nlrse-1.7)

```python
from okean.pipelines.kaping import KAPING
from okean.modules.generative_llms.t5 import T5
from okean.knowledge_base.wikidata import WikidataKG
from okean.modules.entity_linking.refined import ReFinED
from okean.modules.information_retrieval.mpnet import MPNet

text = "Which member of Black Eyed Peas appeared in Poseidon?"

kaping_model = KAPING(
  el_model=ReFinED.from_pretrained(model_path="<PATH_TO_MODEL>", entity_corpus_path="<PATH_TO_CORPUS>"),
  ranking_model=MPNet.from_pretrained(model_path="<PATH_TO_MODEL>"),
  llm_model=T5.from_pretrained(model_path="<PATH_TO_MODEL>"),
  kg=WikidataKG("<PATH_TO_KG>")
)

answer = kaping_model(text)
>> Fergie
```

## Knowledge-Enhanced Information Retrieval (KEIR)
```python
from okean.pipelines.keir import KEIR
from okean.modules.information_retrieval.dpr import DPR
from okean.modules.entity_linking.refined import ReFinED
from okean.preprocessing.doc_transformation import EntityDisambiguation

text = "What year did Michael Jordan win his first NBA championship?"

keir_model = KEIR(
  el_model=ReFinED.from_pretrained(model_path="<PATH_TO_MODEL>", entity_corpus_path="<PATH_TO_CORPUS>"),
  ir_model=DPR.from_pretrained(model_path="<PATH_TO_MODEL>", document_corpus_path="<PATH_TO_CORPUS>"),
  doc_transformer=EntityDisambiguation()
)

relevant_docs = keir_model(text)
```

## Training Toolkit
```python
from okean.modules.entity_linking.refined import ReFinED
from okean.datasets.entity_linking import EntityLinkingDataset
from okean.trainers.entity_linking.refined import ReFinEDTrainer

train_set = EntityLinkingDataset("<PATH_TO_DATASET>")
dev_set = EntityLinkingDataset("<PATH_TO_DATASET>")

el_model = ReFinED.from_pretrained(model_path="<PATH_TO_MODEL>", entity_corpus_path="<PATH_TO_CORPUS>")
el_trainer = ReFinEDTrainer(el_model)
el_trainer.train(train_set=train_set, dev_set=dev_set)

el_trainer.save_pretrained("<PATH_TO_SAVE>")
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

el_model = ReFinED.from_pretrained(model_path="<PATH_TO_MODEL>", entity_corpus_path="<PATH_TO_CORPUS>")
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
