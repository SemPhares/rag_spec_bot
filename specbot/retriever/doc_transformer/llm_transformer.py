from utils.usefull import List
from config import ModelConfig
from model_api.llamacpp_model import llamacpp_from_pretrained

class llama_embeder():
    
    
    def __init__(self):
        """
        
        """
        self.embeder = llamacpp_from_pretrained(ModelConfig.EMBEDDING_MODEL_REPO_ID,
                                                        ModelConfig.EMBEDDING_MODEL_FILENAME)

    
    def create_embedding(self, text:str) -> List[float]:
        """
        """
        emb = self.embeder.create_embedding(text) # type: ignore
        emb: List[float] = [e.embedding for e in emb.data] # type: ignore
        return emb
        

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # embeddings = [self.create_embedding(text) for text in texts]
        embeddings = self.embeder.embed(texts) # type: ignore
        return embeddings # type: ignore


    def embed_query(self, text: str) -> List[float]:
        """Generate query embeddings using FastEmbed.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        # query_embeddings = self.create_embedding(text) 
        query_embeddings = self.embeder.embed([text]) # type: ignore
        return query_embeddings    # type: ignore
