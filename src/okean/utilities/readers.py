import json
from typing import Any, Dict, List


def read_entity_corpus(path: str):
    entity_corpus = {}
    with open(path, "r") as f:
        for line in f:
            entity = json.loads(line)
            entity_corpus[entity["id"]] = {"name": entity["name"], "desc": entity["desc"]}
    return entity_corpus


def load_entity_corpus(load_path: str) -> List[Dict[str, Any]]:
        corpus_contents = []
        with open(load_path, "r") as f:
            for line in f:
                entity = json.loads(line)
                corpus_contents.append(entity)
        return corpus_contents