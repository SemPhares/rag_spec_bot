# Import necessary libraries
import tempfile
import streamlit as st
from streamlit_chat import message
from utils import logger
from prompter.prompt import build_rag_prompt
from doc_loader.loader import CustomeLoader
from model_api import ask_llmcpp, ask_ollama
from retriever.doc_transformer import others_transformer
from retriever.vectorstore import FAISS, retrieve_docs


# Set the title for the Streamlit app
st.title("SPECBOT 🤖")
logger.info("App started")

# Create a file uploader in the sidebar
uploaded_files = st.sidebar.file_uploader("Upload File", accept_multiple_files=True)

# Create a selectbox for file extension
file_extension = st.sidebar.selectbox(
    "Select File Extension",
    ("pdf", "docx", "txt", "csv",  "json", "xml", "html", "md", "pptx", "xls", "xlsx") )


# Handle file upload
if len(uploaded_files) > 0:
    logger.info(f"File uploaded: {len(uploaded_files)}")
    filename_list = [file.name for file in uploaded_files]
    # log the file names
    logger.info(f"File names: {filename_list}")
    # Write liste of temporary files
    tempfile_path_list = []
    for file in uploaded_files:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(file.getvalue())
        tempfile_path_list.append(temp_file.name)

    # log the temporary file names
    logger.info(f"Temp file names: {tempfile_path_list}")

    docs = CustomeLoader(filename_list=filename_list,
                         tempfile_path_list=tempfile_path_list).load()

    logger.info(f"Number of documents: {len(docs)}")
    embedder = others_transformer.ollama_embeder
    # Create a FAISS vector store and save embeddings
    vector_store = FAISS.from_documents(docs, embedder)

    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize messages
    if 'generated' not in st.session_state:
        # file name to display comma separated
        file_name_to_display = ", ".join(filename_list)
        st.session_state['generated'] = ["Hello ! Ask me about " + file_name_to_display + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["hi ! 👋"]

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Question your pdf file", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            retrieved_docs = retrieve_docs(user_input, vector_store)
            prompt = build_rag_prompt(user_input, retrieved_docs)
            logger.info(f"Prompt: {prompt}")
            output = ask_ollama(prompt)
            logger.info(f"Output: {output}")
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat history
    if st.session_state['generated']:
        with response_container:

            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="personas")
                message(st.session_state["generated"][i], key=str(i), avatar_style="micah")