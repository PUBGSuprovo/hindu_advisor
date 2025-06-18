# vector_db.py
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings # Use the generic Embeddings type
import os
import logging

def load_chroma_db(embedding: Embeddings, directory: str = "/tmp/chroma_db"): # Default directory changed
    """
    Loads an existing ChromaDB vector store from the specified directory.

    Args:
        embedding (Embeddings): The embedding function used to create the vector store.
                                            Must be the same one used during persistence.
        directory (str): The path to the directory where the ChromaDB is persisted.
                         This will now typically be '/tmp/chroma_db' after extraction.

    Returns:
        Chroma: An instance of the loaded ChromaDB vector store.

    Raises:
        FileNotFoundError: If the specified ChromaDB directory does not exist.
        Exception: For other errors during loading.
    """
    # In this new setup, the 'directory' will be '/tmp/chroma_db' which is created/populated
    # by the download_and_extract_db function in fastapi_app.py
    if not os.path.exists(directory):
        logging.error(f"ChromaDB directory '{directory}' not found. "
                      "Ensure it has been downloaded and extracted by the app startup process.")
        raise FileNotFoundError(f"ChromaDB directory '{directory}' not found. "
                                "Please ensure the database is available locally after startup.")
    try:
        db = Chroma(persist_directory=directory, embedding_function=embedding)
        logging.info(f"Successfully loaded ChromaDB from '{directory}'.")
        return db
    except Exception as e:
        logging.error(f"Error loading ChromaDB from '{directory}': {e}", exc_info=True)
        raise
