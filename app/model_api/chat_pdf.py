from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough

from langchain.document_loaders.pdf import PyPDFLoader
from langchain.vectorstores.faiss import FAISS
from langchain_community.chat_models.ollama import ChatOllama
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None
    DB_PATH = 'vectorstore/db_faiss'

    def __init__(self):
        self.model = ChatOllama(model="mistral",
                       temperature=0.5,
                       top_k=0.3,)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        template = (
                "<s> [INST] Combine the chat history and follow up question into "
                "a standalone question. Chat History: {chat_history}"
                "Follow up question: {question} [/INST] </s> "
            )
        self.prompt = PromptTemplate.from_template(template)


    def ingest_pdf_file(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        docs = self.text_splitter.split_documents(docs)
        docs = filter_complex_metadata(docs)

        embeddings = FastEmbedEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        # Create a FAISS vector store and save embeddings
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(self.DB_PATH)
    
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        # combine_docs_chain = StuffDocumentsChain(...)

        # question_generator_chain = ({"question": RunnablePassthrough()}
        #               | self.prompt
        #               | self.model
        #               | StrOutputParser()) 

        # self.chain = ConversationalRetrievalChain(
        #         combine_docs_chain= combine_docs_chain,
        #         retriever=self.retriever,
        #         question_generator=question_generator_chain)

        # Create a conversational chain
        self.chain = ConversationalRetrievalChain.from_llm(llm=self.model, 
                                                      retriever=self.retriever)


    # Function for conversational chat
    def ask_chat_pdf(self, query:str, chat_history):
        if not self.chain:
            return "Please, add a PDF document first."
        
        result = self.chain.invoke({"question": query, "chat_history": chat_history})
        chat_history.append((query, result["answer"]))
        return result["answer"]
    

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
