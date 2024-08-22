from typing import List
from config.model_config import ModelConfig
from model_api.llamacpp_model import llamacpp_embedder

import torch
from sentence_transformers import SentenceTransformer

from google import generativeai as genai
from config.model_config import ApiConfig
genai.configure(api_key=ApiConfig.GOOGLE_API_KEY)


class llama_cpp_embeder():
    
    def __init__(self):
        """
        
        """
        self.embeder = llamacpp_embedder(ModelConfig.EMBEDDING_MODEL_PATH)


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



class sentence_embeder():
    
    model_name: str = "intfloat/multilingual-e5-large-instruct"


    def __init__(self):
        """
        
        """
        # load torch device

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"  # Use MPS if available on macOS with Apple Silicon
        else:
            self.device = "cpu"
            
        self.embeder = SentenceTransformer(model_name_or_path= self.model_name, 
                                           device= self.device)


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return [self.embeder.encode(t).tolist() for t in texts]


    def embed_query(self, query: str) -> List[float]:
        """Generate query embeddings using FastEmbed.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embeder.encode(query).tolist()



class gemini_embeder():
    
    
    def __init__(self):
        """
        
        """
        self.model = "models/text-embedding-004"
        

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings : dict = genai.embed_content(model=self.model,
                                  content=texts)
        
        return embeddings["embedding"]


    def embed_query(self, text: str) -> List[float]:
        """Generate query embeddings using FastEmbed.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        # query_embeddings = self.create_embedding(text) 
        query_embeddings = genai.embed_content(model=self.model,
                                  content=text)
        return query_embeddings["embedding"]

