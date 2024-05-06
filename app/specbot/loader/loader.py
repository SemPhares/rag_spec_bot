from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

# from langchain_community.document_loaders.merge import MergedDataLoader
# loader_all = MergedDataLoader(loaders=[loader_web, loader_pdf])

from pathlib import Path
from typing import Iterator
from spliter import text_splitter
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document



class CustomeLoader(BaseLoader):
    """An example document loader that reads a file line by line."""
    accepted_extension = {'.docx':Docx2txtLoader, 
                          '.doc':Docx2txtLoader, 
                          '.pdf':PyPDFLoader, 
                          '.pptx':UnstructuredPowerPointLoader, 
                          '.txt':TextLoader,
                          '.xls':UnstructuredExcelLoader,
                          '.xlsx':UnstructuredExcelLoader}


    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path


    def __split_ext(self,):
        """
        """
        path = Path(self.file_path)
        name = path.stem
        extension = path.suffix
        return name, extension
    

    def __return_rigth_loader(self):
        """
        
        """
        name, extension = self.__split_ext()
        if not extension in self.accepted_extension:
            raise NameError(f"L'extension {extension} n'est pas prise en charge")
        else:
            return self.accepted_extension[extension]


    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        loader = self.__return_rigth_loader()
        documents = loader(self.file_path).load()

        return text_splitter.split_documents(documents)
    
