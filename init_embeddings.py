# init_embeddings.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Changed import
from langchain.text_splitter import CharacterTextSplitter
import logging
import google.generativeai as genai # Needed for genai.configure
from dotenv import load_dotenv # For local environment variables

load_dotenv() # Load .env file for local development

# Configure logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_all_pdfs(folder_path: str) -> list:
    """
    Loads all PDF documents from a specified folder.

    Args:
        folder_path (str): The path to the folder containing PDF files.

    Returns:
        list: A list of loaded Langchain Document objects.
    """
    all_docs = []
    if not os.path.exists(folder_path):
        logging.error(f"PDF folder '{folder_path}' not found. Please ensure it exists and contains PDFs.")
        raise FileNotFoundError(f"PDF folder '{folder_path}' not found.")

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            path = os.path.join(folder_path, file_name)
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                all_docs.extend(docs)
                logging.info(f"✅ Loaded: {file_name} ({len(docs)} chunks)")
            except Exception as e:
                logging.error(f"❌ Error loading PDF '{file_name}': {e}")
    if not all_docs:
        logging.warning(f"No PDF documents were loaded from '{folder_path}'. Ensure PDFs are present.")
    return all_docs

def main():
    """
    Main function to load PDFs, split them into chunks, generate embeddings,
    and persist the vector store using ChromaDB.
    """
    pdf_dir = "pdfs"
    output_dir = "db"

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logging.critical("GEMINI_API_KEY is not set. Please set it in your environment or .env file.")
        return

    # Configure Gemini for embedding initialization
    genai.configure(api_key=gemini_api_key)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: '{output_dir}'")

    logging.info(f"Starting embedding process. Loading PDFs from '{pdf_dir}'...")
    raw_docs = load_all_pdfs(pdf_dir)

    if not raw_docs:
        logging.error("No documents to process. Exiting.")
        return

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)
    logging.info(f"📚 Total chunks after splitting: {len(chunks)}")

    if not chunks:
        logging.error("No chunks generated from documents. Check splitter configuration or document content. Exiting.")
        return

    try:
        embedding = GoogleGenerativeAIEmbeddings(model="embedding-001", google_api_key=gemini_api_key) # Changed here
        logging.info("Initialized GoogleGenerativeAIEmbeddings model: embedding-001")
    except Exception as e:
        logging.critical(f"Failed to initialize embedding model: {e}. Exiting.")
        return

    logging.info(f"Creating and persisting vector store in '{output_dir}'...")
    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=output_dir
        ).persist()
        logging.info(f"✅ Vector store successfully created and persisted in '{output_dir}'")
    except Exception as e:
        logging.critical(f"Failed to create or persist ChromaDB: {e}. Ensure permissions are correct.")

if __name__ == "__main__":
    main()
