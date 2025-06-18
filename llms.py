# llms.py
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import logging # Ensure logging is imported if used directly here, though config.py sets it up

def init_gemini_llm(api_key: str):
    """
    Initializes and configures the Google Gemini LLM.

    Args:
        api_key (str): Your Google Gemini API key.

    Returns:
        GoogleGenerativeAI: An instance of the configured Gemini LLM.
    """
    if not api_key:
        logging.error("Gemini API Key is missing. Cannot initialize Gemini LLM.")
        raise ValueError("GEMINI_API_KEY must be provided to initialize Gemini LLM.")
    try:
        genai.configure(api_key=api_key)
        # Using gemini-1.5-flash for faster responses, temperature for creativity
        return GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.5)
    except Exception as e:
        logging.error(f"Error initializing Gemini LLM: {e}")
        raise

def init_embeddings():
    """
    Initializes and returns HuggingFace embeddings for text.

    Uses 'all-MiniLM-L6-v2' model for efficient sentence embeddings.

    Returns:
        HuggingFaceEmbeddings: An instance of the HuggingFace embeddings model.
    """
    try:
        # HuggingFaceEmbeddings will handle downloading the model if not present.
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}, # Consider "cuda" if a GPU is available
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        logging.error(f"Error initializing HuggingFace embeddings: {e}")
        raise