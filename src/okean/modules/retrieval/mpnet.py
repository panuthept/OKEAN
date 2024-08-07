from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from okean.modules.retrieval.baseclass import Retrieval, RetrievalConfig



@dataclass
class MPNetConfig(RetrievalConfig):
    pass


class MPNet(Retrieval):
    @property
    def available_models(self):
        return [
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-mpnet-base-v2",
        ]

    @staticmethod
    def _load_from_hub(hub_path: str):
        # Load config
        config = MPNetConfig(
            pretrained_model_name=hub_path,
        )
        # Load model
        path_to_models = hf_hub_download(hub_path, "pytorch_model.bin")
        return config, path_to_models
    

if __name__ == "__main__":
    model = MPNet.from_pretrained(
        model_name_or_path="sentence-transformers/all-mpnet-base-v2",
        text_corpus_path="./data/text_corpus/test.jsonl",
        precomputed_text_corpus_path="./data/precomputed_text_corpus/test",
    )
    model.precompute_corpus(
        save_path="./data/precomputed_text_corpus/test", batch_size=8, verbose=True, overwrite=True
    )
    
    texts = [
        "That is a happy person",
        "I have a dog",
        "The weather is nice today",
    ]
    response = model(texts, batch_size=8)
    print(response.runtimes)
    for passage in response.passages:
        print(passage.text)
        print("-" * 100)
        for relavant_passage in passage.relevant_passages:
            print(f"  - {relavant_passage.text} ({relavant_passage.confident:.2f})")
        print("=" * 100)
    print("*" * 100)