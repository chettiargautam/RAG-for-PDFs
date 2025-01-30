import os
import pdfplumber
import sqlite3
from typing import List, Tuple
import argparse

from models.embeddings import generate_embeddings


class PDFProcessingPipeline:
    """
    `PDFProcessingPipeline` class to process PDF files, extract text, chunk it, and optionally push to a database.
    """
    def __init__(self, pdf_paths: List[str], push_to_db: bool = False):
        """
        Initializes the pipeline with the list of PDF paths and a flag to push results to a database.
        :param pdf_paths: List of paths to the PDF files.
        :param push_to_db: Flag to determine if the extracted chunks should be pushed to a database.
        """
        self.pdf_paths = pdf_paths
        self.push_to_db = push_to_db

    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Validates the PDF: checks if the file exists, if it's a valid PDF, 
        and ensures the file is not empty.
        :param pdf_path: Path to the PDF file
        :return: True if the PDF is valid, False otherwise
        """
        if not os.path.exists(pdf_path):
            print(f"File does not exist: {pdf_path}")
            return False
        if not pdf_path.lower().endswith('.pdf'):
            print(f"Invalid file type (not a PDF): {pdf_path}")
            return False
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) == 0:
                    print(f"Empty PDF: {pdf_path}")
                    return False
        except Exception as e:
            print(f"Failed to open PDF: {pdf_path}. Error: {e}")
            return False
        return True

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts text from a given PDF.
        :param pdf_path: Path to the PDF file
        :return: Extracted text from the PDF
        """
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"Error extracting text from PDF: {pdf_path}. Error: {e}")
        return text

    def chunk_text(self, text: str, chunk_size: int = 500, overlap_size: int = 50) -> List[str]:
        """
        Chunk the text into smaller parts with optional overlap.
        :param text: The text content extracted from the PDF
        :param chunk_size: Size of each chunk
        :param overlap_size: The number of words to overlap between chunks
        :return: A list of text chunks
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def push_chunks_to_db(self, pdf_path: str, chunks: List[str]):
        """
        Pushes the extracted chunks and metadata to an SQLite database, 
        including an empty field for embeddings that will be filled later.
        :param pdf_path: Path to the PDF file
        :param chunks: List of text chunks
        """
        conn = sqlite3.connect('db/knowledgebase.db')
        c = conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS chunks
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_path TEXT,
                    chunk_number INTEGER,
                    chunk_text TEXT,
                    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding BLOB)''')

        for idx, chunk in enumerate(chunks, start=1):
            embeddings = generate_embeddings([chunk])[0]
            c.execute('''INSERT INTO chunks (pdf_path, chunk_number, chunk_text, embedding)
                        VALUES (?, ?, ?, ?)''', (pdf_path, idx, chunk, embeddings.tobytes()))

        conn.commit()
        conn.close()

    def process_pdfs(self) -> List[Tuple[str, List[str]]]:
        """
        Process all the PDFs in the list and return their extracted chunks.
        If the push_to_db flag is True, the chunks will be inserted into the database.
        :return: A list of tuples where each tuple contains a PDF path and its text chunks
        """
        processed_data = []
        for pdf_path in self.pdf_paths:
            if self.validate_pdf(pdf_path):
                text = self.extract_text_from_pdf(pdf_path)
                chunks = self.chunk_text(text)
                if self.push_to_db:
                    self.push_chunks_to_db(pdf_path, chunks)
                processed_data.append((pdf_path, chunks))
        return processed_data


def main():
    parser = argparse.ArgumentParser(description="Process PDF files to extract and chunk text.")
    parser.add_argument('pdf_paths', metavar='P', type=str, nargs='+', 
                        help="Paths to the PDF files (separated by space).")
    parser.add_argument('--db', action='store_true', help="Flag to push results to a database.")
    parser.add_argument('--display', action='store_true', help="Flag to display the extracted chunks.")
    args = parser.parse_args()

    pipeline = PDFProcessingPipeline(pdf_paths=args.pdf_paths, push_to_db=args.db)
    
    results = pipeline.process_pdfs()

    if args.display:
        for pdf_path, chunks in results:
            print(f"PDF: {pdf_path}")
            for idx, chunk in enumerate(chunks, start=1):
                print(f"Chunk {idx}: {chunk}")
            print()


if __name__ == "__main__":
    main()