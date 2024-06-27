from specbot.config import ModelConfig, GlobalConfig
from langchain_community.embeddings.ollama import OllamaEmbeddings

ollama_embeder = OllamaEmbeddings(model=ModelConfig.EMBEDDING_MODEL_NAME, 
                                  num_gpu=GlobalConfig.NUM_GPU)

#intfloat/multilingual-e5-large