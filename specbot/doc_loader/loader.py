from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader


from langchain.vectorstores.utils import filter_complex_metadata

# from langchain_community.document_loaders.merge import MergedDataLoader
# loader_all = MergedDataLoader(loaders=[loader_web, loader_pdf])

import tempfile
from utils.usefull import List
from .spliter import text_splitter
from .lodaer_utils import logger, timer
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from .lodaer_utils import extract_everithing_from_doc, caption_single_image


class CustomeLoader(BaseLoader):
    """An example document loader that reads a file line by line."""
    ACCEPTED_EXTENSION = {'docx':Docx2txtLoader, 
                          'doc':Docx2txtLoader, 
                          'pdf':PyPDFLoader, 
                          'pptx':UnstructuredPowerPointLoader, 
                          'txt':TextLoader,
                          'xls':UnstructuredExcelLoader,
                          'xlsx':UnstructuredExcelLoader}
    
    IMAGES_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif']  # Define the IMAGES_EXTENSIONS variable
    
    # Create a temporary directory
    TEMP_DIR = tempfile.TemporaryDirectory()


    def __init__(self,
                 filename_list: List[str],
                 tempfile_path_list: List[str]) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.filename_list = filename_list
        self.tempfile_path_list = tempfile_path_list
        self.zip_name_path = zip(self.filename_list, self.tempfile_path_list)


    def extract_file_extension(self, file_name) -> str:
        """
        Return the extension of the file

        Args:
            file_name: The name of the file
        
        Returns:
            str: The extension of the file
        """
        return file_name.split('.')[-1]
    

    def return_rigth_loader(self, 
                            file_extension:str) -> BaseLoader:
        """
        Return the right loader for the files
        
        Args:
            file_extension: The extension of the file

        Returns:
            BaseLoader: The right loader for the file        
        """
        if not file_extension in self.ACCEPTED_EXTENSION:
            # log warning extension not supported
            logger.warning(f"Extension {self.extract_file_extension} not supported, using default pdf loader.")
            return self.ACCEPTED_EXTENSION['pdf']
        else:
            return self.ACCEPTED_EXTENSION[file_extension]
        

    @timer
    def lazy_load(self) -> List[Document]:  # <-- Does not take any arguments
        """
        A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        # all documents list initialization
        all_documents = []

        # Load all the documents
        for filename, path in self.zip_name_path:
            file_extension = self.extract_file_extension(filename)

            if file_extension in self.IMAGES_EXTENSIONS:
                logger.info(f"Image file identified: {filename}")
                document = caption_single_image(path)
                all_documents.append(document)
                continue

            loader = self.return_rigth_loader(file_extension)
            # liste de documents
            documents:list = loader(file_path = path).load() # type: ignore
            all_documents.extend(documents)
            other_documents = extract_everithing_from_doc(path, self.TEMP_DIR) # type: ignore
            all_documents.extend(other_documents)

        all_documents = text_splitter.split_documents(all_documents)
        # Filter out documents with complex metadata
        all_documents = filter_complex_metadata(all_documents)
        return all_documents
    
