# KALM
A Universal Framework for Knowledge-Augmented Language Model (KALM) Applications

## Components

### Entity Linking
```python
from kalm.el.blink import BLINK

text = "What year did Michael Jordan win his first NBA championship?"

blink_model = BLINK.from_pretrained(model_path="./models/el/blink", entity_corpus_path="./data/entity_corpus/wikipedia_en")
pred = blink_model(text)
>> Doc(text="What year did Michael Jordan win his first NBA championship?", spans=[Span(start=14, end=28, surface_form="Michael Jordan", entity="Michael Jordan")])

blink_model = BLINK.from_pretrained(model_path="./models/el/blink", entity_corpus_path="./data/entity_corpus/wikidata")
pred = blink_model(text)
>> Doc(text="What year did Michael Jordan win his first NBA championship?", spans=[Span(start=14, end=28, surface_form="Michael Jordan", entity="Q41421")])
```
