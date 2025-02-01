import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

<context>
{context}
</context>

Answer the following question:

{question}
"""

def get_documents_from_directory(directory_path: str) -> list:
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter()
    return splitter.split_documents(documents)

def get_local_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text")

def create_vectorstore_from_documents(documents: list, embeddings: OllamaEmbeddings) -> Chroma:
    return Chroma.from_documents(documents=documents, embedding=embeddings)

def get_chat_model() -> ChatOllama:
    return ChatOllama(model="llama3.2:1b")

def retrieve_similar_documents(vectorstore: Chroma, question: str) -> list:
    return vectorstore.similarity_search(question)

def format_documents(documents: list) -> str:
    return "\n\n".join(document.page_content for document in documents)

def rag_chatbot_from_vectorstore(question: str, vectorstore: Chroma) -> str:
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    retriever = vectorstore.as_retriever()
    qa_chain = (
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | rag_prompt
        | get_chat_model()
        | StrOutputParser()
    )
    return qa_chain.invoke(question)

st.set_page_config(page_title="RAG Chatbot", layout="wide")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

page = st.sidebar.selectbox("Select Page", ["Upload PDFs", "Ask Questions"])

if os.path.exists("uploaded_pdfs"):
    for file in os.listdir("uploaded_pdfs"):
        os.remove(os.path.join("uploaded_pdfs", file))
    os.rmdir("uploaded_pdfs")

if page == "Upload PDFs":
    st.title("Upload PDF Files")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
    if st.button("Process PDFs"):
        if uploaded_files:
            os.makedirs("uploaded_pdfs", exist_ok=True)
            for file in uploaded_files:
                with open(os.path.join("uploaded_pdfs", file.name), "wb") as f:
                    f.write(file.getbuffer())
            
            st.info("Processing PDFs and creating vector store...")
            documents = get_documents_from_directory("uploaded_pdfs")
            embeddings = get_local_embeddings()
            st.session_state.vectorstore = create_vectorstore_from_documents(documents, embeddings)
            st.success("Vector store created successfully!")
        else:
            st.warning("Please upload PDFs before processing.")

elif page == "Ask Questions":
    st.title("Ask Questions to the RAG Chatbot")
    if st.session_state.vectorstore is None:
        st.error("No vector store found. Please upload PDFs first.")
    else:
        question = st.text_input("Enter your question")
        if st.button("Get Answer") and question:
            response = rag_chatbot_from_vectorstore(question, st.session_state.vectorstore)
            st.write("### Answer:")
            st.write(response)