import os
import shutil
import numpy as np
from copy import deepcopy
from typing import List, Dict, Optional
from okean.data_types.basic_types import Passage
from okean.modules.retrieval.baseclass import Retriever
from okean.packages.bgem3_package.bge_m3 import BGEM3FlagModel
# from FlagEmbedding import BGEM3FlagModel


class BGEM3(Retriever):
    def __init__(
            self,
            model_path: str = "BAAI/bge-m3", 
            corpus_path: Optional[str] = None, 
            device: Optional[str] = None,
            use_fp16: bool = True,
    ):
        self.corpus_contents = []
        self.corpus_embeddings = None

        if corpus_path is not None and os.path.exists(corpus_path):
            print(f"Loading corpus from {corpus_path}.")
            self.load_corpus(corpus_path)

        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16, device=device)
        self.device = self.model.device

    def save_corpus(self, corpus_path: str):
        os.makedirs(corpus_path, exist_ok=True)
        np.save(os.path.join(corpus_path, "embeddings.npy"), self.corpus_embeddings, allow_pickle=True)
        with open(os.path.join(corpus_path, "contents.txt"), "w") as f:
            f.write("\n".join(self.corpus_contents))

    def load_corpus(self, corpus_path: str):
        self.corpus_embeddings = np.load(os.path.join(corpus_path, "embeddings.npy"), allow_pickle=True).item()
        with open(os.path.join(corpus_path, "contents.txt"), "r") as f:
            self.corpus_contents = f.read().splitlines()

    def _encoding(self, texts: List[str], batch_size: int = 8) -> Dict[str, np.ndarray]:
        return self.model.encode(
            texts, 
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )

    def queries_encoding(self, texts: List[str], batch_size: int = 8) -> Dict[str, np.ndarray]:
        return self._encoding(texts, batch_size=batch_size)
    
    def corpus_encoding(self, texts: List[str], batch_size: int = 8) -> Dict[str, np.ndarray]:
        return self._encoding(texts, batch_size=batch_size)
    
    def build_corpus(
            self, 
            corpus_path: str, 
            texts: List[str], 
            batch_size: int = 8, 
            remove_existing: bool = False, 
            skip_existing: bool = True
    ):
        if os.path.exists(corpus_path) and not remove_existing:
            if skip_existing:
                print(f"Corpus already exists at {corpus_path}. Set `remove_existing=True` to overwrite.")
                return
            raise FileExistsError(f"Corpus already exists at {corpus_path}. Set `skip_existing=True` to skip or `remove_existing=True` to overwrite.")
        
        if os.path.exists(corpus_path) and remove_existing:
            print(f"Removing existing corpus at {corpus_path}.")
            self.corpus_contents = []
            self.corpus_embeddings = None
            shutil.rmtree(corpus_path)

        self.corpus_contents = texts
        self.corpus_embeddings = self.corpus_encoding(texts, batch_size=batch_size)
        self.save_corpus(corpus_path)

    def __call__(
            self, 
            texts: List[str]|str = None, 
            passages: List[Passage]|Passage = None,
            batch_size: int = 8,
            dense_weight: float = 0.4,
            sparse_weight: float = 0.2,
            colbert_weight: float = 0.4,
            k: int = 10,
    ) -> List[Passage]:
        # Cast `texts` to `passages` if `passages` is not provided
        if passages is None:
            assert texts is not None, "Either `text` or `passages` must be provided."
            if isinstance(texts, list):
                passages = [Passage(text=t) for t in texts]
            else:
                passages = [Passage(text=texts)]
        # Ensure that `passages` is a list of `Passage` objects
        if not isinstance(passages, list):
            passages = [passages]

        if self.corpus_embeddings is None:
            raise ValueError("Corpus not built. Use `build_corpus` method to build the corpus.")

        queries: List[str] = [passage.text for passage in passages]
        queries_embeddings = self.queries_encoding(queries, batch_size=batch_size)

        dense_scores = np.array(queries_embeddings["dense_vecs"] @ self.corpus_embeddings["dense_vecs"].T)
        sparse_scores = np.array([[self.model.compute_lexical_matching_score(query_lexical_weights, corpus_lexical_weights) for corpus_lexical_weights in self.corpus_embeddings["lexical_weights"]] for query_lexical_weights in queries_embeddings["lexical_weights"]])
        colbert_scores = np.array([[self.model.colbert_score(query_colbert_vecs, corpus_colbert_vecs) for corpus_colbert_vecs in self.corpus_embeddings["colbert_vecs"]] for query_colbert_vecs in queries_embeddings["colbert_vecs"]])
        bgem3_scores = (dense_weight * dense_scores) + (sparse_weight * sparse_scores) + (colbert_weight * colbert_scores)

        indicess = np.argsort(bgem3_scores, axis=1)[:, ::-1][:, :k]
        scoress = np.take_along_axis(bgem3_scores, indicess, axis=1)

        passages = deepcopy(passages)
        for passage, scores, indices in zip(passages, scoress, indicess):
            passage.relevant_passages = [{"relevant_passage": [Passage(text=self.corpus_contents[idx], confident=score) for score, idx in zip(scores, indices)]}]
        return passages


if __name__ == "__main__":
    retriever = BGEM3(corpus_path="./corpus/BGEM3", use_fp16=False)

    corpus = [
        "Stacy Ann 'Fergie' Ferguson (born March 27, 1975) is an American singer, songwriter, rapper and actress. She first achieved chart success as part of the hip hop group the Black Eyed Peas. Her debut solo album, The Dutchess (2006), saw commercial success and spawned three Billboard Hot 100 number one singles: 'London Bridge', 'Glamorous', and 'Big Girls Don't Cry'.", 
        "Jaime Luis Gomez[2] (born July 14, 1975),[3] better known by the stage names Taboo,[4] or Taboo Nawasha is an American rapper, singer, songwriter, actor, DJ, and comic book writer, best known as a member of the musical group Black Eyed Peas.",
        "Fergie has additionally appeared as an actress in various films, such as Poseidon (2006), Grindhouse (2007), and Nine (2009). She launched her first fragrance, Outspoken, under Avon in May 2010 and has since released four more fragrances. Her footwear line, Fergie Footwear, was launched in 2009.",
        "Black Eyed Peas (also known as The Black Eyed Peas) is an American musical group consisting of rappers will.i.am, apl.de.ap and Taboo. The group's lineup during the height of their popularity in the 2000s[5] featured Fergie, who replaced Kim Hill in 2002. Originally an alternative hip hop group, they subsequently refashioned themselves as a more marketable pop-rap act.[4] Although the group was founded in Los Angeles in 1995, it was not until the release of their third album Elephunk in 2003 that they achieved high record sales.",
        "Poseidon is a 2006 American action disaster film directed and co-produced by Wolfgang Petersen. It is the third film adaptation of Paul Gallico's 1969 novel The Poseidon Adventure, and a loose remake of the 1972 film. It stars Kurt Russell, Josh Lucas and Richard Dreyfuss with Emmy Rossum, Jacinda Barrett, Mike Vogel, Mía Maestro, Jimmy Bennett and Andre Braugher in supporting roles. It was produced and distributed by Warner Bros. in association with Virtual Studios. It had a simultaneous release in IMAX format. It was released on May 12, 2006, and it was criticized for its script but was praised for its visuals and was nominated at the 79th Academy Awards for Best Visual Effects.[2] It grossed $181.7 million worldwide on a budget of $160 million; however, after the costs of promotion and distribution, Warner Bros. lost $70–80 million on the film, making it a box-office bomb as a result.",
        "Michael Jeffrey Jordan (born February 17, 1963), also known by his initials MJ,[9] is an American businessman and former professional basketball player. He played fifteen seasons in the National Basketball Association (NBA) between 1984 and 2003, winning six NBA championships with the Chicago Bulls. He was integral in popularizing basketball and the NBA around the world in the 1980s and 1990s,[10] becoming a global cultural icon.[11] His profile on the NBA website states, 'By acclamation, Michael Jordan is the greatest basketball player of all time.'",
    ]
    query = "Which member of Black Eyed Peas appeared in Poseidon?"

    retriever.build_corpus(corpus_path="./corpus/BGEM3", texts=corpus, remove_existing=False)
    results = retriever(query)
    print(results)