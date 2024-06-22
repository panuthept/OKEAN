import torch
import numpy as np
from torch import Tensor
from torch.functional import F
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
from okean.modules.retrieval.baseclass import DenseRetriever, IndexConfig


class MPNet(DenseRetriever):
    def __init__(
            self, 
            model_path: str = "microsoft/mpnet-base", 
            index_config: Optional[IndexConfig] = None, 
            corpus_path: Optional[str] = None, 
            device: Optional[str] = None,
            use_fp16: bool = True,
    ):
        if index_config is None: index_config = IndexConfig(ndim=768, metric="ip", dtype="f32")
        super().__init__(index_config, corpus_path)
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self.model.to(self.device)
        if use_fp16: self.model.half()

    def _average_pooling(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _encoding(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for i in range(0, len(texts), batch_size):
            inputs = self.tokenizer(texts[i:i + batch_size], max_length=512, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._average_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                embeddings = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()
                all_embeddings[i:i + batch_size] = embeddings
        return all_embeddings

    def queries_encoding(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        return self._encoding(texts, batch_size=batch_size)
    
    def corpus_encoding(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        return self._encoding(texts, batch_size=batch_size)
    

if __name__ == "__main__":
    retriever = MPNet(corpus_path="./corpus/MPNet")

    corpus = [
        "Stacy Ann 'Fergie' Ferguson (born March 27, 1975) is an American singer, songwriter, rapper and actress. She first achieved chart success as part of the hip hop group the Black Eyed Peas. Her debut solo album, The Dutchess (2006), saw commercial success and spawned three Billboard Hot 100 number one singles: 'London Bridge', 'Glamorous', and 'Big Girls Don't Cry'.", 
        "Jaime Luis Gomez[2] (born July 14, 1975),[3] better known by the stage names Taboo,[4] or Taboo Nawasha is an American rapper, singer, songwriter, actor, DJ, and comic book writer, best known as a member of the musical group Black Eyed Peas.",
        "Fergie has additionally appeared as an actress in various films, such as Poseidon (2006), Grindhouse (2007), and Nine (2009). She launched her first fragrance, Outspoken, under Avon in May 2010 and has since released four more fragrances. Her footwear line, Fergie Footwear, was launched in 2009.",
        "Black Eyed Peas (also known as The Black Eyed Peas) is an American musical group consisting of rappers will.i.am, apl.de.ap and Taboo. The group's lineup during the height of their popularity in the 2000s[5] featured Fergie, who replaced Kim Hill in 2002. Originally an alternative hip hop group, they subsequently refashioned themselves as a more marketable pop-rap act.[4] Although the group was founded in Los Angeles in 1995, it was not until the release of their third album Elephunk in 2003 that they achieved high record sales.",
        "Poseidon is a 2006 American action disaster film directed and co-produced by Wolfgang Petersen. It is the third film adaptation of Paul Gallico's 1969 novel The Poseidon Adventure, and a loose remake of the 1972 film. It stars Kurt Russell, Josh Lucas and Richard Dreyfuss with Emmy Rossum, Jacinda Barrett, Mike Vogel, Mía Maestro, Jimmy Bennett and Andre Braugher in supporting roles. It was produced and distributed by Warner Bros. in association with Virtual Studios. It had a simultaneous release in IMAX format. It was released on May 12, 2006, and it was criticized for its script but was praised for its visuals and was nominated at the 79th Academy Awards for Best Visual Effects.[2] It grossed $181.7 million worldwide on a budget of $160 million; however, after the costs of promotion and distribution, Warner Bros. lost $70–80 million on the film, making it a box-office bomb as a result.",
        "Michael Jeffrey Jordan (born February 17, 1963), also known by his initials MJ,[9] is an American businessman and former professional basketball player. He played fifteen seasons in the National Basketball Association (NBA) between 1984 and 2003, winning six NBA championships with the Chicago Bulls. He was integral in popularizing basketball and the NBA around the world in the 1980s and 1990s,[10] becoming a global cultural icon.[11] His profile on the NBA website states, 'By acclamation, Michael Jordan is the greatest basketball player of all time.'",
    ]
    query = "Which member of Black Eyed Peas appeared in Poseidon?"

    retriever.build_corpus(corpus_path="./corpus/MPNet", texts=corpus, remove_existing=False)
    results = retriever(query)
    print(results)