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
uploaded_file = st.sidebar.file_uploader("Upload File", type="pdf")

# Handle file upload
if uploaded_file:
    logger.info(f"File uploaded: {uploaded_file}")
    file_name = uploaded_file.name
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

        logger.info(f"File name: {tmp_file_path}")

        docs = CustomeLoader(file_path=tmp_file_path,
                             file_bytes=tmp_file).load()
        logger.info(f"Number of documents: {len(docs)}")
        embeddings = others_transformer.FastEmbedEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        # Create a FAISS vector store and save embeddings
        vector_store = FAISS.from_documents(docs, embeddings)

    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

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
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")