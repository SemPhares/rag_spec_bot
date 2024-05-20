from llama_cpp import Llama
from typing import List
import numpy as np


class llama_embeder():
    
    model_name: str = ""
    """
    """

    repo_id: str = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF"
    """
    repo_id
    """


    def __init__(self):
        """
        
        """
        self.embeder = Llama(embedding=True).from_pretrained(
                    repo_id=self.repo_id)
        

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings: List[np.ndarray]

        # if self.doc_embed_type == "passage":
        #     embeddings = self._model.passage_embed(texts)
        # else:
        embeddings = self.embeder.create_embedding(texts)
        return [e.tolist() for e in embeddings]


    def embed_query(self, text: str) -> List[float]:
        """Generate query embeddings using FastEmbed.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        query_embeddings: np.ndarray = next(self.embeder.create_embedding(text))
        return query_embeddings.tolist()
