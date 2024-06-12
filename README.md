# KALM - Knowledge-Augmented Language Model
A Universal Framework for Knowledge-Augmented Language Model (KALM) Applications

## Example
```python
from kalm.lm import LUKE
from kalm.el.blink import BLINK

doc = "What year did Michael Jordan win his first NBA championship?"

blink_model = BLINK.from_pretrained(model_path="<PATH_TO_MODEL>", entity_corpus_path="<PATH_TO_CORPUS>")
doc = blink_model(doc)
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
