from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')


def generate_embeddings(chunks: List[str]) -> np.ndarray:
    """
    Generate embeddings for the provided text chunks.
    :param chunks: List of text chunks.
    :return: Array of embeddings.
    """
    embeddings = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
    return embeddings


if __name__ == "__main__":
    chunks = ["This is a sample sentence."]
    embeddings = generate_embeddings(chunks)
    print(embeddings[0])
    print(embeddings.shape)