# vector_db.py
from langchain_community.vectorstores import Chroma
# Changed import from HuggingFaceEmbeddings to BaseEmbeddings as it's more generic
from langchain_core.embeddings import Embeddings # Use the generic Embeddings type
import os
import logging

def load_chroma_db(embedding: Embeddings, directory: str = "db"): # Changed type hint
    """
    Loads an existing ChromaDB vector store from the specified directory.

    Args:
        embedding (Embeddings): The embedding function used to create the vector store.
                                            Must be the same one used during persistence.
        directory (str): The path to the directory where the ChromaDB is persisted.
                         This will be the local path where the DB is downloaded (e.g., from HF).

    Returns:
        Chroma: An instance of the loaded ChromaDB vector store.

    Raises:
        FileNotFoundError: If the specified ChromaDB directory does not exist.
        Exception: For other errors during loading.
    """
    # In a deployment scenario, this 'db' directory would be downloaded from Hugging Face.
    # We assume 'db' exists locally (after download) when this function is called.
    if not os.path.exists(directory):
        logging.error(f"ChromaDB directory '{directory}' not found. "
                      "Ensure it has been downloaded and extracted locally.")
        raise FileNotFoundError(f"ChromaDB directory '{directory}' not found. "
                                "Please ensure the database is available locally.")
    try:
        db = Chroma(persist_directory=directory, embedding_function=embedding)
        logging.info(f"Successfully loaded ChromaDB from '{directory}'.")
        return db
    except Exception as e:
        logging.error(f"Error loading ChromaDB from '{directory}': {e}")
        raise
