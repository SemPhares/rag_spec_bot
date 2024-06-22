from config import Config
from utils.log import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter


config = {
    "chunk_size" : 1024,
    "chunk_overlap" : 100,
}

text_splitter = RecursiveCharacterTextSplitter(chunk_size = int(Config.CHUNCK_SIZE), 
                                               chunk_overlap = int(Config.CHUNK_OVERLAP))
