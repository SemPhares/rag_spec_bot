from config import GlobalConfig
from utils.log import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter



text_splitter = RecursiveCharacterTextSplitter(chunk_size = int(GlobalConfig.CHUNCK_SIZE), 
                                               chunk_overlap = int(GlobalConfig.CHUNK_OVERLAP))