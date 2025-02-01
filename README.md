# RAG for PDFs: A Retrieval-Augmented Generation Chatbot

## Introduction

This project implements a Retrieval-Augmented Generation (RAG) chatbot that can extract and answer questions based on content from uploaded PDF documents. Using **Langchain**, **Ollama**, and **Chroma**, the system builds a document retrieval pipeline that first processes PDF files into embeddings, stores them in a vector store, and then uses those embeddings to retrieve the most relevant content for a given query. A language model (Ollama) is then used to generate a response based on the retrieved documents.

### Langchain

Langchain is used throughout the pipeline to handle document loading, splitting, embedding creation, vector store management, and prompt generation. The key features of Langchain that are leveraged include:

- **Document Loaders**: Langchain provides loaders such as `PyPDFDirectoryLoader` to easily load and process PDFs.
- **Text Splitters**: The `RecursiveCharacterTextSplitter` is used to split large documents into smaller, manageable chunks that can be processed more efficiently.
- **Embeddings and Vector Stores**: Langchain supports the creation of embeddings and the use of a vector store (via Chroma) to store these embeddings, enabling fast similarity searches.
- **Chat Models**: Langchain's integration with Ollama models allows for easy inference of local models for generating responses.

### The Logic Behind Document Creation and Extraction

1. **Directory Loading**: PDFs are uploaded to the application and stored in a specified directory. Using Langchain’s `PyPDFDirectoryLoader`, the documents are loaded from this directory.
2. **Text Splitting**: To prevent model overload, the documents are split into smaller text chunks using `RecursiveCharacterTextSplitter`. This ensures that only manageable portions of text are processed.
3. **Vector Store and Embeddings**: After splitting, the text chunks are converted into vector embeddings using Ollama’s `OllamaEmbeddings`. These embeddings are stored in a **Chroma vector store**, which enables efficient similarity-based retrieval.
4. **Why Ollama?**: Ollama is chosen for its efficient local model inference capabilities, allowing for the creation of high-quality embeddings and text generation (answering questions based on the context).

### RAG Process Flow

1. **Document Preprocessing**: First, PDFs are uploaded and processed into documents.
2. **Retrieval**: When a user asks a question, the system performs a similarity search over the vector store to retrieve the most relevant document chunks.
3. **Generation**: The retrieved documents are passed into the Ollama model, which generates a natural language response based on the content.
4. **Final Answer**: The response is returned to the user.

---

## Installation and Setup

### Dependencies

Before running the project, make sure you have the necessary dependencies installed. Use the following `requirements.txt` to install them:

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

### Setting Up Ollama Locally

This project uses **Ollama** for local model inference. To set it up, follow these steps:

1. **Install Ollama**: You can download Ollama from [Ollama's website](https://ollama.com/).
2. **Download Models**:
   - **`nomic-embed-text`**: For embedding generation.
   - **`llama3.2:1b`**: For text generation (answering questions).

Once Ollama is installed, use the following commands to download the models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

### Running the Project Locally

1. **Run `rag.py`**:
   - `rag.py` contains all the core logic for document processing, embedding generation, and question answering.
   - To test the functionality locally, you can run `rag.py` directly by specifying the path to your PDFs:

```bash
python rag.py
```

   - Ensure that your PDFs are placed in the `data/` directory and the path is correct in the script.

2. **Run the Streamlit App**:
   - The Streamlit app is a user-friendly interface to upload PDFs and ask questions.
   - To start the app, simply run the following command:

```bash
streamlit run app.py
```

   - The Streamlit app will open in your browser. You can upload PDFs, process them, and ask questions to the RAG chatbot.

---

## Main Functions in `rag.py`

Here’s a breakdown of the main functions in `rag.py`, along with sample usages and expected inputs/outputs:

1. **`get_documents_from_directory(directory_path)`**

   - **Description**: Loads PDFs from the specified directory and splits them into smaller chunks.
   - **Sample Usage**:
   
   ```python
   documents = get_documents_from_directory('/path/to/pdf/directory')
   ```
   
   - **Input**: Path to a directory containing PDF files.
   - **Output**: A list of document chunks (each chunk is an object with a `page_content` attribute containing text).
   - **Example Output**:
   
   ```python
   [
     {'page_content': "Introduction to Python programming..."},
     {'page_content': "In this chapter, we will cover..."}
   ]
   ```

2. **`get_local_embeddings()`**

   - **Description**: Returns the Ollama embeddings model for generating document embeddings.
   - **Sample Usage**:
   
   ```python
   embeddings = get_local_embeddings()
   ```
   
   - **Output**: Ollama embeddings model.
   
3. **`create_vectorstore_from_documents(documents, embeddings)`**

   - **Description**: Converts the documents into embeddings and stores them in a vector store.
   - **Sample Usage**:
   
   ```python
   vectorstore = create_vectorstore_from_documents(documents, embeddings)
   ```
   
   - **Input**: A list of document chunks and the embeddings model.
   - **Output**: A Chroma vector store containing the document embeddings.
   - **Example Output**:
   
   ```python
   <Chroma object with embedded documents>
   ```

4. **`rag_chatbot_from_vectorstore(question, vectorstore)`**

   - **Description**: Given a question and a vector store, retrieves relevant documents and generates a response using the RAG model.
   - **Sample Usage**:
   
   ```python
   response = rag_chatbot_from_vectorstore(question, vectorstore)
   ```
   
   - **Input**: A question string and a Chroma vector store.
   - **Output**: A string containing the model-generated response.
   - **Example Output**:
   
   ```python
   "To start your full-stack developer journey, you should first learn..."
   ```

5. **`retrieve_similar_documents(pdf_directory_path, question)`**

   - **Description**: Retrieves documents from the specified directory based on a similarity search with the question.
   - **Sample Usage**:
   
   ```python
   similar_documents = retrieve_similar_documents('/path/to/pdf/directory', "What is the correct way to start your full stack developer journey?")
   ```
   
   - **Input**: A directory path and a question string.
   - **Output**: A list of document chunks relevant to the question.
   - **Example Output**:
   
   ```python
   [
     {'page_content': "Starting your full-stack development journey requires a strong understanding..."},
     {'page_content': "The first step is to learn JavaScript and how to build web applications..."}
   ]
   ```

---

## Using the Streamlit App

To use the Streamlit app:

1. **Upload PDFs**: Navigate to the "Upload PDFs" page in the app. Upload multiple PDF files, and then click the **"Process PDFs"** button.
2. **Ask Questions**: Once the PDFs are processed, go to the "Ask Questions" page, enter your query, and click **"Get Answer"**.
3. **View Response**: The chatbot will retrieve relevant documents and generate a response.

- Run the Streamlit application through the command line in the following manner:

    ```bash
    streamlit run app.py    
    ```

---

## Conclusion

This project demonstrates a practical implementation of a Retrieval-Augmented Generation (RAG) chatbot that can efficiently process and answer questions based on the content of PDFs. By leveraging Langchain for document processing and Ollama for local model inference, the system is capable of providing accurate and contextually relevant answers. With the Streamlit interface, users can easily upload documents, create embeddings, and interact with the RAG chatbot.