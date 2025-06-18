# llms.py
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Import for Gemini Embeddings
import google.generativeai as genai
import logging

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

def init_embeddings(api_key: str): # Modified to accept API key
    """
    Initializes and returns Google Gemini embeddings for text.

    Args:
        api_key (str): Your Google Gemini API key.

    Returns:
        GoogleGenerativeAIEmbeddings: An instance of the Gemini embeddings model.
    """
    if not api_key:
        logging.error("Gemini API Key is missing. Cannot initialize Gemini Embeddings.")
        raise ValueError("GEMINI_API_KEY must be provided to initialize Gemini Embeddings.")
    try:
        return GoogleGenerativeAIEmbeddings(model="embedding-001", google_api_key=api_key)
    except Exception as e:
        logging.error(f"Error initializing Gemini Embeddings: {e}")
        raise
