# KALM - Knowledge-Augmented Language Model
A Universal Framework for Knowledge-Augmented Language Model (KALM) Applications

## Inference
```python
from kalm.lm.luke import LUKE
from kalm.el.refined import ReFinED

doc = "What year did Michael Jordan win his first NBA championship?"

el_model = ReFinED.from_pretrained(model_path="<PATH_TO_MODEL>", entity_corpus_path="<PATH_TO_CORPUS>")
doc = el_model(doc)
>> Doc(
  text="What year did Michael Jordan win his first NBA championship?",
  spans=[
    Span(
      start=14,
      end=28,
      surface_form="Michael Jordan",
      entity=Entity(
        id="Q41421",
        name="Michael Jeffrey Jordan",
        desc="American basketball player and businessman (born 1963)",
        confident=1.0,
      )
    )
  ]
)

lm_model = LUKE.from_pretrained(model_path="<PATH_TO_MODEL>")
sentence_emb = lm_model(doc)
```

## Training
```python
from kalm.el.refined import ReFinED
from kalm.dataset import EntityLinkingDataset

train_set = EntityLinkingDataset("<PATH_TO_DATASET>")
dev_set = EntityLinkingDataset("<PATH_TO_DATASET>")

el_model = ReFinED.from_pretrained(model_path="<PATH_TO_MODEL>", entity_corpus_path="<PATH_TO_CORPUS>")
el_model.train(train_set=train_set, dev_set=dev_set)

el_model.save_pretrained("<PATH_TO_SAVE>")
```
