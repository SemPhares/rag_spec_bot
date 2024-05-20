from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings

fast_embeder = FastEmbedEmbeddings()

ollama_embeder = OllamaEmbeddings()

open_embeder = OpenAIEmbeddings()
#intfloat/multilingual-e5-large