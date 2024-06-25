import base64
from glob import glob
from config import ModelConfig
from utils.log import logger
from utils.usefull import timer
from langchain_core.documents import Document
from unstructured.partition.auto import partition
from langchain_core.output_parsers import StrOutputParser
from model_api.llamacpp_model import llamacpp_from_pretrained, llama_cpp_image_input, llamacpp_for_caption


@timer
def extrcat_elements_from(file_path, 
                          image_output_dir:str):
    """
    
    """
    raw_elements = partition(file_path,
                             strategy='hi_res',
                             extract_images_in_pdf=True,
                             chunking_strategy="by_title",
                             extract_image_block_output_dir=image_output_dir)
    return raw_elements


# Create a dictionary to counts of each type of element extrcat from the document
def elements_text_and_tables(elements):
    """
    Count the number of each type of element in a list of elements.
    """
    element_count = {}
    tables = []
    texts = []
    others = []
    for element in elements:
        element_type = str(type(element))

        if "unstructured.documents.elements.Table" in element_type:
           tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in element_type:
           texts.append(str(element))
        else:
            others.append(str(element))
                
        if element_type in element_count:
            element_count[element_type] += 1
        else:
            element_count[element_type] = 1
    
    logger.info(f"Nombre d'Element extraits: {element_count}")
    return texts, tables, others


def convert_to_str(elements):
    """
    """
    str_list = []
    for i in elements:
        if isinstance(i, str) and i != "":
            str_list.append(i)
        elif hasattr(i, "text") and i.text != "": # type: ignore
            str_list.append(i.text) # type: ignore
    return str_list


def summarize_tables(tables, 
                     file_path:str = "") -> list[Document]:
    """
    """
    summarize_tables = []
    tables = convert_to_str(tables)

    logger.info(f"Tables to summarize: {len(tables)}")

    model = llamacpp_from_pretrained(ModelConfig.PHI3_INSTRUCT_REPO_ID,
                                     ModelConfig.PHI3_INSTRUCT_FILENAME)
    
    if tables:
        for table in tables:
            # Prompt
            prompt_text = f"""You are an assistant tasked with summarizing tables and text. \
                            Give a concise summary of the table or text. Table or text chunk: {table} """
            output = model.create_completion(prompt_text)
            output = output["choices"][0]["text"] # type: ignore
            table_doc = Document(page_content=StrOutputParser().parse(output),
                        metadata = {"type": "table", "source": ModelConfig.PHI3_MODEL_NAME, "file_path": file_path})
            summarize_tables.append(table_doc)

        logger.info(f"Tables summarized: {len(summarize_tables)}")
        logger.info(f"Tables summarized first: {summarize_tables[0].page_content}")
    return summarize_tables


def texts_to_documents(texts, file_path:str = "") -> list[Document]:
    """
    """
    texts = convert_to_str(texts)
    metadata = {"type": "text",  "source": 'unstructured', "file_path": file_path}
    return [Document(page_content=text, metadata=metadata) for text in texts]
    

def encode_image(image_path):
    ''' Getting the base64 string '''
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def caption_images(image_bs4) -> str:
    """
    """
    model = llamacpp_from_pretrained(ModelConfig.IMAGE_MODEL_REPO_ID,
                                     ModelConfig.IMAGE_MODEL_FILENAME)
    
    response = model.create_chat_completion(
        messages = [
            {
                "role": "user",
                "content": [
                    {"type" : "text", "text": "Describe the image in detail. Be specific about graphs, such as bar plots."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_bs4}" } }
                ]
            }
        ],
        temperature=0.2,
    )
    caption:str = response["choices"][0]["message"]['content'] # type: ignore
    return caption


def caption_single_image(image_path:str) -> Document:
    """
    """
    logger.info(f"Processing image: {image_path}")
    image_bs4 = encode_image(image_path)
    caption_input = llama_cpp_image_input(input=ModelConfig.EXTRACTED_IMAGE_PROMPT,
                                          repo_id=ModelConfig.IMAGE_MODEL_REPO_ID,
                                          filename=ModelConfig.IMAGE_MODEL_FILENAME,
                                          model_name=ModelConfig.IMAGE_MODEL_NAME,
                                          image_bs4=image_bs4)
    
    img_cpation = llamacpp_for_caption(caption_input)
    image_doc = Document(page_content=img_cpation.response,
                         metadata = {"type": "image", "source": ModelConfig.IMAGE_MODEL_NAME, "file_path": image_path})
    return image_doc


@timer
def summarize_iamges(images_dir:str) -> list[Document]:
    """
    """
    images_caption = []
    # glob all the images in the directory png or jpg
    
    images_list = list(glob(f"{images_dir}/*.jpg"))
    logger.info(f"Images list: {images_list}")
    for image_path in images_list:
        image_doc = caption_single_image(image_path)
        images_caption.append(image_doc)
    return images_caption


@timer
def extract_everithing_from_doc(file_path:str,
                                image_output_dir:str):
    """
    """
    # Extract elements
    elements = extrcat_elements_from(file_path, image_output_dir)
    # Extract text and tables
    texts, tables, others = elements_text_and_tables(elements)
    # Summarize tables
    tables_docs = summarize_tables(tables, file_path)
    # Text to documents
    text_docs = texts_to_documents(texts, file_path)
    # Other to documents
    others_docs = texts_to_documents(others, file_path)
    # Summarize images
    image_docs = summarize_iamges(image_output_dir)
    # Combine all the documents into one list
    all_documents = text_docs + tables_docs + image_docs + others_docs

    return all_documents
