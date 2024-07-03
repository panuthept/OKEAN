AVAILABLE_PRETRAINED_IR_MODELS = {
  "E5": [
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-large-instruct",
    "intfloat/e5-small",
    "intfloat/e5-base",
    "intfloat/e5-large",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
  ],
  "MPNet": [
      "sentence-transformers/all-mpnet-base-v2"
  ],
  "BGEm3": [
    "BAAI/bge-m3",
  ],
}

DENSE_RETRIEVAL_CONFIGS = {
    "E5": {
        "max_query_length": 512,
        "max_passage_length": 512,
        "promt": {
            "query": "query: {text}",
            "passage": "passage: {text}",
        },
        "pooling_method": "average",
        "normalize_embeddings": True,
        "similarity_distance": "dot",
    },
    "Default": {
        "max_query_length": 512,
        "max_passage_length": 512,
        "promt": None,
        "pooling_method": "average",
        "normalize_embeddings": True,
        "similarity_distance": "dot",
    },
}