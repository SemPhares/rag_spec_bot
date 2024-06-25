import torch
import numpy as np
from specbot.utils.usefull import List
from sentence_transformers import SentenceTransformer



class sentence_embeder():
    
    model_name: str = "intfloat/multilingual-e5-large-instruct"
    """
    model_name
    """


    def __init__(self):
        """
        
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeder = SentenceTransformer(model_name_or_path= self.model_name, device= self.device)


    def embed(self, input_query:str):
        """
        
        """
        output = self.embeder.encode([input_query])[0]
        return  np.array([output])        



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
        embeddings = [self.embed(text) for text in texts]
        return [e.tolist() for e in embeddings]


    def embed_query(self, text: str) -> List[float]:
        """Generate query embeddings using FastEmbed.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        query_embeddings: np.ndarray = self.embed(text)
        return query_embeddings.tolist()
