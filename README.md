# KALM
A Universal Framework for Knowledge-Augmented Language Model (KALM) Applications

## Example
```python
from kalm.lm import LUKE
from kalm.el.blink import BLINK

doc = "What year did Michael Jordan win his first NBA championship?"

blink_model = BLINK.from_pretrained(model_path="./models/el/blink", entity_corpus_path="./data/entity_corpus/wikidata")
doc = blink_model(doc)
>> Doc(text="What year did Michael Jordan win his first NBA championship?", spans=[Span(start=14, end=28, surface_form="Michael Jordan", entity="Q41421")])

lm_model = LUKE.from_pretrained(model_path="./models/lm/luke", entity_corpus_path="./data/entity_corpus/wikidata")
sentence_emb = lm_model(doc)
```
