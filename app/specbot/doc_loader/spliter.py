# from app.specbot.config import Config
from langchain_text_splitters import RecursiveCharacterTextSplitter


config = {
    "chunk_size" : 1024,
    "chunk_overlap" : 100,
}

text_splitter = RecursiveCharacterTextSplitter(chunk_size = config["chunk_size"], 
                                               chunk_overlap = config['chunk_overlap'])
