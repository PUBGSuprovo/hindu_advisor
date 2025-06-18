# fastapi_app.py

import os
import json
import logging
import zipfile
import requests
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware # For in-memory session history
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage # For explicit message types
from langchain_community.chat_message_histories import ChatMessageHistory # For in-memory history

# Google Sheets logging (Ensure gspread and oauth2client are installed in requirements.txt)
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Download & Extract Vector DB (from your old code) ---
def download_and_extract_db():
    # IMPORTANT: Ensure this URL points to a ZIP file containing your ChromaDB's 'db' folder
    # Example: A zip of your 'db' folder for 'Dyno1307/hindu_db' would be
    # https://huggingface.co/datasets/Dyno1307/hindu_db/resolve/main/db.zip if you zipped it.
    # If your dataset ISN'T zipped, you must adapt this or use the build.sh method.
    # Assuming 'Dyno1307/hindu_db' is now a zipped file containing the 'db' directory inside.
    url = "https://huggingface.co/datasets/Dyno1307/hindu_db/resolve/main/db.zip" # ADJUST THIS IF YOUR DB IS NOT ZIPPED AT THIS EXACT URL
    zip_path = "/tmp/db.zip"
    extract_path = "/tmp/chroma_db" # This is where ChromaDB will be extracted

    # Check if DB already exists to avoid re-downloading on every startup (important for Render restarts)
    if os.path.exists(os.path.join(extract_path, "index")): # Check for a common ChromaDB subdirectory
        logging.info("✅ Chroma DB already exists at /tmp/chroma_db, skipping download.")
        return

    try:
        logging.info(f"⬇️ Downloading Chroma DB zip from: {url}")
        response = requests.get(url, stream=True, timeout=300) # Increased timeout for large files
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Optional: log progress if needed, but can be verbose
                    # logging.debug(f"Downloaded {downloaded_size}/{total_size} bytes")

        logging.info(f"📦 Extracting zip to {extract_path}...")
        # Create directory if it doesn't exist
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        logging.info("✅ Vector DB extracted successfully.")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"❌ Network or HTTP error downloading Vector DB: {req_err}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to download Vector DB: {req_err}")
    except zipfile.BadZipFile:
        logging.error("❌ Downloaded file is not a valid zip file.", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to extract Vector DB: Corrupted zip file.")
    except Exception as e:
        logging.error(f"❌ General error downloading or extracting Vector DB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to prepare Vector DB: {e}")


# --- Environment Setup ---
load_dotenv() # Load .env variables for local development
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Renamed from GOOGLE_CREDENTIALS_JSON to GOOGLE_CREDS_JSON as per your old code
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")
# Secret key for FastAPI sessions (important for production!)
FASTAPI_SECRET_KEY = os.getenv("FASTAPI_SECRET_KEY", "a_very_strong_random_secret_key_change_this_for_production")

if not GEMINI_API_KEY:
    logging.critical("❌ GEMINI_API_KEY not found in environment variables. Application will not function.")
    raise ValueError("GEMINI_API_KEY not set. Please set it in your environment or .env file.")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress noisy logs from specific langchain modules if needed
logging.getLogger('langchain_community.chat_message_histories.in_memory').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING) # Suppress http client logs

# --- Google Sheets Setup (from your old code) ---
# Global variables for Google Sheets client and sheet enable flag
gs_sheet = None
gs_sheet_enabled = False
if GOOGLE_CREDS_JSON:
    try:
        creds_dict = json.loads(GOOGLE_CREDS_JSON)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            creds_dict,
            ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive",
             "https://www.googleapis.com/auth/spreadsheets"]
        )
        gs_client = gspread.authorize(creds)
        # IMPORTANT: Replace "Diet Suggest Logs" with your ACTUAL Google Sheet Name for Hindu Advisor project
        gs_sheet = gs_client.open("Hindu Scripture Advisor Logs").sheet1 # Assumes you have a sheet named "Hindu Scripture Advisor Logs" with at least one sheet
        gs_sheet_enabled = True
        logging.info("✅ Google Sheets connected for logging.")
    except json.JSONDecodeError:
        logging.warning("⚠️ GOOGLE_CREDS_JSON is not a valid JSON string. Google Sheets logging disabled.")
    except Exception as e:
        logging.warning(f"⚠️ Google Sheets connection failed: {e}. Logging to sheet disabled.", exc_info=True)
else:
    logging.info("ℹ️ GOOGLE_CREDS_JSON not set. Skipping Google Sheets logging.")


# --- FastAPI App Init ---
app = FastAPI(
    title="Hindu Scripture Advisor API", # Project-specific title
    description="A backend API for Hindu scripture and spiritual guidance using RAG and LLMs.", # Project-specific description
    version="0.2.1", # Increment version for this fix
)
# CORS configuration for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust this to specific origins in production, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Session middleware for conversational history (in-memory)
app.add_middleware(SessionMiddleware, secret_key=FASTAPI_SECRET_KEY)

# --- Import Local Modules (ADAPTED for your current project's module names) ---
from utils import ( # All query analysis and utility functions are in utils.py now
    extract_spiritual_concept, extract_life_problem, extract_scripture_source,
    clean_response_text, clean_suggestions, contains_table_request,
    merge_groq_and_rag_answers # This function is in utils.py
)
from llms import init_gemini_llm, init_embeddings # llms.py handles LLM and embeddings initialization
from vector_db import load_chroma_db # vector_db.py handles loading ChromaDB
from qa_chain import create_qa_chain, get_session_history_in_memory # qa_chain.py for QA chain and in-memory history factory
from groq_api import cached_groq_answers # groq_api.py for Groq integrations
from prompts import scripture_prompt, merge_prompt_default, merge_prompt_table # prompts.py for prompt templates

# --- LLM & Vector DB Setup (App Startup Logic) ---
# Download and extract the database at app startup (first time it runs)
# This replaces the build.sh step
download_and_extract_db()

# Initialize LLM and Embeddings
llm_gemini = init_gemini_llm(GEMINI_API_KEY)
embedding_model = init_embeddings(GEMINI_API_KEY) # init_embeddings now requires API key

# Setup Vector DB
db_instance = None
try:
    # IMPORTANT: Use the extract_path from download_and_extract_db for loading ChromaDB
    db_instance = load_chroma_db(embedding_model, directory="/tmp/chroma_db")
    count = len(db_instance.get()['documents'])
    logging.info(f"✅ Vector DB initialized with {count} documents.")
except Exception as e:
    logging.error(f"❌ Vector DB init error: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="Vector DB initialization failed. Check /tmp/chroma_db and permissions.")

# Setup LangChain RAG and Conversational Components
rag_chain = None
conversational_rag_chain = None
try:
    # create_qa_chain requires llm and retriever
    rag_chain = create_qa_chain(llm_gemini, db_instance.as_retriever(), scripture_prompt) # Pass scripture_prompt
    
    # get_session_history_in_memory is the simple in-memory factory
    conversational_rag_chain = rag_chain.with_message_history(get_session_history_in_memory)
    
    logging.info("✅ LangChain RAG and Conversational Chains initialized successfully.")
except Exception as e:
    logging.error(f"❌ LangChain QA Chain setup error: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="LangChain QA Chain initialization failed.")


# --- Pydantic Schema for Request Body ---
class ChatRequest(BaseModel):
    query: str
    session_id: str = None # Optional session ID from client, can be used to override FastAPI session

# --- Chat Endpoint ---
@app.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    user_query = chat_request.query
    client_session_id = chat_request.session_id
    
    # Use client_session_id if provided, otherwise generate/retrieve from FastAPI's session
    session_id = client_session_id or request.session.get("session_id")
    if not session_id:
        session_id = f"session_{os.urandom(8).hex()}"
        request.session["session_id"] = session_id # Store new session ID in FastAPI session

    logging.info(f"📩 Query: '{user_query}' | Session: {session_id}")

    # Query analysis (from utils.py)
    is_greeting_flag = contains_table_request(user_query) # Reusing contains_table_request for simplicity, review original logic
    # The original "old code" used `is_greeting` and `is_formatting_request`
    # Let's use the ones from utils.py which you confirmed are present
    is_greeting_flag = extract_spiritual_concept(user_query).lower() == "general" and not contains_table_request(user_query) # Simple greeting logic based on extract
    is_formatting_flag = contains_table_request(user_query) # Use this to check for table request
    wants_table_flag = contains_table_request(user_query) # This is redundant if is_formatting_flag handles table requests

    # Use the parameter extraction functions from utils.py
    extracted_spiritual_concept = extract_spiritual_concept(user_query)
    extracted_life_problem = extract_life_problem(user_query)
    extracted_scripture_source = extract_scripture_source(user_query)
    
    # Retrieve chat history for this session (using the in-memory store)
    # The `get_session_history_in_memory` is from qa_chain.py
    chat_history_messages = get_session_history_in_memory(session_id).messages

    response_text = ""

    # --- Intent-Based Routing ---
    if is_greeting_flag and not wants_table_flag: # Refined greeting logic
        response_text = "Namaste! How can I assist you with Hindu scriptures today?"
    elif wants_table_flag and len(chat_history_messages) > 1: # Attempt to reformat previous AI message as table
        logging.info("Handling formatting request for previous answer.")
        last_ai_message_content = None
        # Find the last AI message in history
        for msg in reversed(chat_history_messages):
            if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                last_ai_message_content = msg.content
                break

        if last_ai_message_content:
            try:
                # Use the merge function to reformat the last AI message
                # Provide dummy suggestions as we are only reformatting the RAG answer
                dummy_suggestions = {"llama": "", "mixtral": "", "gemma": ""}
                
                response_text = await merge_groq_and_rag_answers(
                    llm_gemini, # Use llm_gemini for reformatting
                    last_ai_message_content, # The content to reformat
                    dummy_suggestions,
                    extracted_spiritual_concept,
                    extracted_life_problem,
                    extracted_scripture_source,
                    format_as_table=True # Force table format
                )
            except Exception as e:
                logging.error(f"❌ Reformat error: {e}", exc_info=True)
                response_text = "I encountered an issue reformatting the previous response. Please try again."
        else:
            response_text = "I can only re-format a previous answer. Please ask for guidance first!"
    else: # Default to RAG + Groq + Merge pipeline for task-oriented queries
        logging.info("Handling RAG + Groq + Merge pipeline.")
        rag_output_content = "No answer from knowledge base."
        try:
            rag_result = await conversational_rag_chain.ainvoke({
                "question": user_query,
                "spiritual_concept": extracted_spiritual_concept, # Pass params to chain if prompt uses them
                "life_problem": extracted_life_problem,
                "scripture_source": extracted_scripture_source
            }, config={
                "configurable": {"session_id": session_id} # This links to the in-memory session
            })
            
            rag_output_content = rag_result.get("answer", "No answer from knowledge base.")
            logging.info(f"✅ RAG Chain Raw Output: {rag_output_content[:200]}...")
        except Exception as e:
            logging.error(f"❌ RAG error during ainvoke: {e}", exc_info=True)
            rag_output_content = "Error while retrieving response from knowledge base."

        groq_suggestions = {}
        try:
            if GROQ_API_KEY:
                groq_suggestions = cached_groq_answers(
                    query=user_query,
                    groq_api_key=GROQ_API_KEY,
                    spiritual_concept=extracted_spiritual_concept,
                    life_problem=extracted_life_problem,
                    scripture_source=extracted_scripture_source
                )
                groq_suggestions = clean_suggestions(groq_suggestions)
                logging.info("✅ Groq suggestions fetched.")
            else:
                logging.warning("GROQ_API_KEY not set. Skipping Groq suggestions.")
        except Exception as e:
            logging.error(f"❌ Groq error during suggestions: {e}", exc_info=True)
            groq_suggestions = {"llama": "Error", "mixtral": "Error", "gemma": "Error"}

        final_output_content = "Something went wrong while combining AI suggestions."
        try:
            # Use the merge_groq_and_rag_answers function from utils.py
            final_output_content = await merge_groq_and_rag_answers(
                llm_gemini,
                rag_output_content,
                groq_suggestions,
                extracted_spiritual_concept,
                extracted_life_problem,
                extracted_scripture_source,
                format_as_table=wants_table_flag
            )
        except Exception as e:
            logging.error(f"❌ Merge process error: {type(e).__name__}: {e}", exc_info=True)
            final_output_content = "I encountered an issue generating a comprehensive response. Please try again."
            
        response_text = clean_response_text(final_output_content)


    # Add user and AI messages to session history (using the in-memory store)
    get_session_history_in_memory(session_id).add_user_message(HumanMessage(content=user_query))
    get_session_history_in_memory(session_id).add_ai_message(AIMessage(content=response_text))

    # Log to Google Sheet (using direct gspread integration)
    try:
        if gs_sheet_enabled and gs_sheet:
            gs_sheet.append_row([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                session_id,
                user_query,
                response_text # Log the actual response sent to the user
            ])
            logging.info("📝 Logged query and response to Google Sheet.")
    except Exception as e:
        logging.warning(f"⚠️ Google Sheet logging failed: {e}", exc_info=True)

    # Return the response as JSON
    return JSONResponse(content={"answer": response_text, "session_id": session_id})

# --- Simple Health Check Endpoint ---
@app.get("/") # Root endpoint for basic health check
async def root():
    return {"message": "✅ Hindu Scripture Advisor API is running. Use POST /chat to interact."}

@app.get("/health") # Dedicated health check endpoint
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is operational."}

# --- Main entry point for Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    # Render provides the PORT environment variable. Default to 8000 for local testing.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

