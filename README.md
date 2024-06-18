# OKEAN - Open Knowledge Enhancement for Advanced NLP

## Knowledge-Enhanced Information Retrieval (KEIR)
```python
from okean.pipelines.keir import KEIR
from okean.preprocessing.transformation import EntityDisambiguation
from okean.modules.information_retrieval.dpr_package.dpr import DPR
from okean.modules.entity_linking.refined_package.refined import ReFinED

doc = "What year did Michael Jordan win his first NBA championship?"

keir_model = KEIR(
  el_model=ReFinED.from_pretrained(model_path="<PATH_TO_MODEL>", entity_corpus_path="<PATH_TO_CORPUS>"),
  ir_model=DPR.from_pretrained(model_path="<PATH_TO_MODEL>"),
  doc_transformer=EntityDisambiguation()
)

relevant_docs = keir_model(doc)
>> doc = el_model(doc)
>> Doc(
  text="What year did Michael Jordan win his first NBA championship?",
  entities=[
    Span(
      start=14,
      end=28,
      surface_form="Michael Jordan",
      entity=Entity(
        identifier="Q41421",
        confident=1.0,
        metadata={
          "name": "Michael Jeffrey Jordan", 
          "desc": "American basketball player and businessman (born 1963)"
        }
      )
    ),
    Span(
      start=43,
      end=46,
      surface_form="NBA",
      entity=Entity(
        identifier="Q155223",
        confident=1.0,
        metadata={
          "name": "National Basketball Association", 
          "desc": "North American professional men's basketball league"
        }
      )
    )
  ]
)
>> transformed_doc = doc_transformer(doc)
>> Doc(text="What year did Michael Jordan (Michael Jeffrey Jordan) win his first NBA (National Basketball Association) championship?")
>> relevant_docs = ir_model(transformed_doc)
```

## Training Toolkit
```python
from okean.datasets.entity_linking import EntityLinkingDataset
from okean.trainers.entity_linking.refined import ReFinEDTrainer
from okean.modules.entity_linking.refined_package.refined import ReFinED

train_set = EntityLinkingDataset("<PATH_TO_DATASET>")
dev_set = EntityLinkingDataset("<PATH_TO_DATASET>")

el_model = ReFinED.from_pretrained(model_path="<PATH_TO_MODEL>", entity_corpus_path="<PATH_TO_CORPUS>")
el_trainer = ReFinEDTrainer(el_model)
el_trainer.train(train_set=train_set, dev_set=dev_set)

el_trainer.save_pretrained("<PATH_TO_SAVE>")
```

## Evaluation Toolkit
```python
from okean.datasets.entity_linking import EntityLinkingDataset
from okean.benchmarks.entity_linking import EntityLinkingBenchmark
from okean.modules.entity_linking.refined_package.refined import ReFinED

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