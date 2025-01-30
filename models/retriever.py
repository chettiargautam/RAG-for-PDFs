import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from models.embeddings import generate_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model. T5 does not perform very well for this task but it is lightweight making it ideal for testing.
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

def retrieve_chunks(query_embedding, top_k=5):
    """Retrieves the most relevant chunks based on cosine similarity."""
    conn = sqlite3.connect('db/knowledgebase.db')
    c = conn.cursor()
    c.execute("SELECT id, pdf_path, chunk_number, chunk_text, embedding FROM chunks")
    rows = c.fetchall()
    conn.close()
    
    similarities = [
        (1 - cosine(query_embedding, np.frombuffer(row[4], dtype=np.float32)), row)
        for row in rows
    ]
    similarities.sort(reverse=True, key=lambda x: x[0])
    return [sim[1] for sim in similarities[:top_k]]

def generate_prompt(top_chunks, query):
    """Generate the prompt for the LLM based on the retrieved context."""
    context = " ".join([chunk[3] for chunk in top_chunks])
    return f"Context: {context}\nQuery: {query}\nAnswer:"

def get_answer_from_llm(prompt):
    """Generate an answer from the LLM while handling token length issues."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    
    outputs = model.generate(
        inputs,
        max_new_tokens=150,  # Reduce to avoid exceeding model limits
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.95,
        top_k=50
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def rag_chatbot(query):
    """Main function to perform retrieval-augmented generation."""
    query_embedding = generate_embeddings([query])[0]
    top_chunks = retrieve_chunks(query_embedding, top_k=3)
    
    responses = []
    for chunk in top_chunks:
        prompt = generate_prompt([chunk], query)  # Generate prompt using a single chunk
        answer = get_answer_from_llm(prompt)  # Generate answer per chunk
        responses.append({
            "answer": answer,
            "file": chunk[1],
            "chunk_number": chunk[2],
            "retrieved_context": chunk[3]
        })

    return responses


if __name__ == "__main__":
    query = "What is full stack development?"
    results = rag_chatbot(query)
    for res in results:
        print(f"Answer: {res['answer']}")
        print(f"From File: {res['file']}")
        print(f"Chunk Number: {res['chunk_number']}")
        print(f"Retrieved Context: {res['retrieved_context']}\n")