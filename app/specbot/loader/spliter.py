from ..config import Config
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(chunk_size = Config.chunk_size, 
                                               chunk_overlap = Config.chunk_overlap)
