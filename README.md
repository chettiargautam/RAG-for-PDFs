# RAG-for-PDFs

## Introduction

This project implements a Retrieval-Augmented Generation (RAG) system for extracting information from PDFs. The system allows users to upload PDF files, processes the content to generate embeddings, and stores the data in an SQLite database. Later, when users query the system, it retrieves relevant text chunks from the database and generates an answer using the RAG approach with a T5 model.

The system is designed to be lightweight and can handle quick inference without the need for expensive API calls. However, note that the answer quality might not be optimal due to the use of the T5 transformer model, which is not trained on large sequences.

---

## Project Logic

### Core Components

1. **PDF Processing Pipeline**: Extracts text from uploaded PDFs, chunks the content into smaller parts, and optionally stores the processed chunks in an SQLite database.
2. **Embedding Generation**: Uses the `SentenceTransformer` model to generate embeddings for the chunks of text.
3. **Retriever**: Retrieves the most relevant text chunks from the database based on the similarity between the user's query and the stored embeddings.
4. **RAG System**: Combines retrieval of relevant text chunks with text generation using the T5 model to answer the user's query.

---

## Dependencies

### Installing Dependencies

To install the necessary dependencies for the project, run the following command:

```bash
pip install -r requirements.txt
```

This will install all the required libraries including Flask, sentence-transformers, torch, and transformers.

---

## Testing Each Part of the Project Separately

### 1. **Testing the Embedding Function**

The `generate_embeddings` function is used to convert text chunks into embeddings that can be used for similarity comparisons. You can test it by running the following Python code:

```python
from models.embeddings import generate_embeddings

# Sample chunks
chunks = ["This is a sample sentence.", "Here is another example sentence."]

# Generate embeddings
embeddings = generate_embeddings(chunks)

# Print the embeddings
print(embeddings)
```

This will output a NumPy array of embeddings for the provided chunks.

---

### 2. **Testing PDF Processing**

The PDF processing component is responsible for extracting text from PDF files, chunking the text into smaller parts, and storing it in a database (if enabled). To test this, you can create a script or use the existing pipeline functionality.

Here is an example of testing the PDF extraction and chunking:

```python
from pdf_processing.pdf_processing_pipeline import PDFProcessingPipeline

# List of PDF paths
pdf_paths = ["path/to/your/file1.pdf", "path/to/your/file2.pdf"]

# Initialize pipeline with push_to_db flag (set to True to push to DB)
pipeline = PDFProcessingPipeline(pdf_paths=pdf_paths, push_to_db=True)

# Process PDFs
result = pipeline.process_pdfs()

# Display the extracted chunks
for pdf_path, chunks in result:
    print(f"PDF: {pdf_path}")
    for chunk in chunks:
        print(f"Chunk: {chunk}")
```

This will output the text chunks extracted from the PDFs and, if the `push_to_db` flag is set to `True`, will store them in the SQLite database.

---

### 3. **How the Retriever Works**

The retriever retrieves the most relevant text chunks from the database based on the similarity of the query to stored embeddings. You should test this only after the database that contains the chunks has been created. You can test it by running the following:

```python
from models.retriever import retrieve_chunks
from models.embeddings import generate_embeddings

# Example query
query = "What is full stack development?"

# Generate query embedding
query_embedding = generate_embeddings([query])[0]

# Retrieve the top 3 relevant chunks
top_chunks = retrieve_chunks(query_embedding, top_k=3)

# Display the retrieved chunks
for chunk in top_chunks:
    print(f"File: {chunk[1]}, Chunk Number: {chunk[2]}")
```

This will retrieve and display the top 3 chunks based on cosine similarity.

---

### 4. **Overview of the RAG System**

The Retrieval-Augmented Generation (RAG) system combines both the retrieval of relevant chunks and text generation to answer the user's query. This is done in the following way:

- **Embedding Generation**: The user's query is converted into an embedding.
- **Retrieval**: Relevant chunks are fetched from the database using cosine similarity between the query embedding and the chunk embeddings.
- **Generation**: The top retrieved chunks are provided to a T5 model as context, and the model generates an answer.

To test the complete RAG system, you can use the `rag_chatbot` function:

```python
from models.retriever import rag_chatbot

# Example query
query = "What is full stack development?"

# Get responses from RAG system
responses = rag_chatbot(query)

# Display the responses
for res in responses:
    print(f"Answer: {res['answer']}")
    print(f"From File: {res['file']}")
    print(f"Chunk Number: {res['chunk_number']}")
```

This will show the answers generated by the RAG system for your query.

---

## Frontend and Backend Integration

### Frontend Overview

The frontend consists of two pages:

1. **Upload PDF Page**: Allows users to upload multiple PDFs. The files are processed and stored in the database for later use.
2. **Chatbot Page**: Users can enter queries, and the system generates answers using the RAG approach.

The `index.html` page allows users to upload PDFs, and after the files are processed, it redirects to the `chat.html` page, where users can interact with the chatbot.

### Backend Overview

The backend is built with **Flask**. It handles the following:

- File uploads and processing.
- Storing processed PDF chunks and embeddings in the SQLite database.
- Handling user queries and returning relevant responses using the RAG system.

Flask routes are defined to handle file uploads (`/upload`), serve the chatbot page (`/chat`), and handle AJAX requests from the frontend for querying.

---

## Starting the Application

Once all dependencies are installed and the components are tested, you can start the Flask app by running:

```bash
cd /RAG-for-PDFs
python -m frontend.app
```

This will start a development server, and the app will be accessible at `http://127.0.0.1:5000/`.

### Uploading PDFs

1. Open the application in your browser.
2. Go to the "Upload PDF" page.
3. Upload multiple PDF files using the file input.
4. After successful upload and processing, you will be automatically redirected to the "Chat" page.

### Using the Chatbot

On the Chat page:

1. Enter a query in the input box.
2. The system will retrieve relevant chunks from the database and generate an answer using the T5 model.
3. The answer, along with the file name and chunk number, will be displayed.

You can click the "Upload More Files" button to go back to the upload page, or continue chatting on the same page.

---

## Notes

- **Model Limitations**: The T5 model used in this project is a small transformer model (`t5-small`). It is lightweight and fast but may not provide high-quality answers when dealing with complex or lengthy documents, as it is not trained on long sequences. It was chosen for this project because it allows for quick inference without the need for expensive API calls.
- **Success Factors**: While the LLM answer quality is not great, the working of the RAG system is evident if the user provides diverse topics of inputs via PDFs and tests the system by prompting about the different topics in the chat interface.