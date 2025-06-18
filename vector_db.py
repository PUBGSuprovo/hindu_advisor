# vector_db.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # Import for type hinting
import os
import logging

def load_chroma_db(embedding: HuggingFaceEmbeddings, directory: str = "db"):
    """
    Loads an existing ChromaDB vector store from the specified directory.

    Args:
        embedding (HuggingFaceEmbeddings): The embedding function used to create the vector store.
                                            Must be the same one used during persistence.
        directory (str): The path to the directory where the ChromaDB is persisted.

    Returns:
        Chroma: An instance of the loaded ChromaDB vector store.

    Raises:
        FileNotFoundError: If the specified ChromaDB directory does not exist.
        Exception: For other errors during loading.
    """
    if not os.path.exists(directory):
        logging.error(f"ChromaDB directory '{directory}' not found. Please run init_embeddings.py first.")
        raise FileNotFoundError(f"ChromaDB directory '{directory}' not found. Ensure it's created.")
    try:
        db = Chroma(persist_directory=directory, embedding_function=embedding)
        logging.info(f"Successfully loaded ChromaDB from '{directory}'.")
        return db
    except Exception as e:
        logging.error(f"Error loading ChromaDB from '{directory}': {e}")
        raise