from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain_ollama import ChatOllama

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import RunnablePassthrough


RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Remember to answer naturally without any words that are out of general conversation.

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
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )

def create_vectorstore_from_documents(documents: list, embeddings: OllamaEmbeddings) -> Chroma:
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )

def get_chat_model() -> ChatOllama:
    return ChatOllama(
        model="llama3.2:1b"
    )

def format_documents(documents: list) -> str:
    return "\n\n".join(document.page_content for document in documents)

def retrieve_similar_documents(pdf_directory_path: str, question: str) -> list:
    """
    This function retrieves similar documents from a directory of PDFs based on a question.
    
    Sample Usage:
    ------------
    >>> pdf_directory_path = '/path/to/pdf/directory'
    >>> question = "What is the correct way to start your full stack developer journey?"
    >>> response = retrieve_similar_documents(pdf_directory_path, question)
    >>> print(response)
    """
    documents = get_documents_from_directory(pdf_directory_path)
    embeddings = get_local_embeddings()
    vectorstore = create_vectorstore_from_documents(documents, embeddings)
    return vectorstore.similarity_search(question)

def summarize_documents_of_relevant_topics(question: str, pdf_directory_path: str) -> str:
    """
    This function summarizes a document based on a question. The question is used to retrieve similar documents from a directory of PDFs.
    
    Sample Usage:
    ------------
    >>> pdf_directory_path = '/path/to/pdf/directory'
    >>> question = "What is the correct way to start your full stack developer journey?"
    >>> response = summarization_chain(question, pdf_directory_path)
    >>> print(response)
    """
    chat_model = get_chat_model()
    
    prompt = ChatPromptTemplate.from_template(
        "Summarize the main themes in these retrieved docs: {docs}"
    )

    chain = {"docs": format_documents} | prompt | chat_model | StrOutputParser()

    similar_documents = retrieve_similar_documents(pdf_directory_path, question)

    return chain.invoke(similar_documents)

def rag_chatbot_with_retrieval(question: str, pdf_directory_path: str) -> str:
    """
    This function returns a response to a question using the RAG model.
    
    Sample Usage:
    ------------
    >>> pdf_directory_path = '/path/to/pdf/directory'
    >>> question = "What is the correct way to start your full stack developer journey?"
    >>> response = rag_chatbot(question, pdf_directory_path)
    >>> print(response)
    """
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    chain = (
        RunnablePassthrough.assign(
            context=lambda input: format_documents(input["context"])
        )
        | rag_prompt
        | get_chat_model()
        | StrOutputParser()
    )

    similar_documents = retrieve_similar_documents(pdf_directory_path, question)

    return chain.invoke({"context": similar_documents, "question": question})

def rag_chatbot_from_vectorstore(question: str, vectorstore: Chroma) -> str:
    """
    This function returns a response to a question using the RAG model.
    
    Sample Usage:
    ------------
    >>> vectorstore = create_vectorstore_from_documents(documents, embeddings)
    >>> question = "What is the correct way to start your full stack developer journey?"
    >>> response = rag_chatbot_from_vectorstore(question, vectorstore)
    >>> print(response)
    """
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    retriever = vectorstore.as_retriever()

    qa_chain = (
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | rag_prompt
        | get_chat_model()
        | StrOutputParser()
    )

    return qa_chain.invoke(question)


if __name__ == '__main__':
    pdf_directory_path = '/Users/gautamchettiar/Documents/Python Codes/RAG-for-PDFs/data'
    question = "What is the correct way to start your full stack developer journey?"
    vectorstore = create_vectorstore_from_documents(get_documents_from_directory(pdf_directory_path), get_local_embeddings())
    # similar_documents = retrieve_similar_documents(pdf_directory_path, question)
    # summarization_response = summarize_documents_of_relevant_topics(question, pdf_directory_path)
    # rag_response = rag_chatbot_with_retrieval(question, pdf_directory_path)
    response = rag_chatbot_from_vectorstore(question, vectorstore)
    print(response)