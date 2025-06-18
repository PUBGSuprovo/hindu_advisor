# fastapi_app.py

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import logging
import os # Import os for environment variables
from llms import init_gemini_llm, init_embeddings
from vector_db import load_chroma_db
# Corrected import for merge_groq_and_rag_answers and GoogleSheetChatMessageHistory
# Now only import functions/classes directly used by fastapi_app from qa_chain
from qa_chain import create_conversational_chain, get_session_history # get_session_history is now imported here
from utils import (
    extract_spiritual_concept,
    extract_life_problem,
    extract_scripture_source,
    clean_response_text,
    clean_suggestions,
    contains_table_request,
    merge_groq_and_rag_answers # This function is defined in utils.py
)
from groq_api import cached_groq_answers
import uuid
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    session_id: str = None
    format_table: bool = False

# Retrieve API keys and Google Sheet credentials from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON") # Corrected variable name from GOOGLE_CREDENTIALS_JSON to match common practice

if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set. Gemini LLM will not be initialized.")
    # In production, you might want to raise an exception here to prevent startup
    # raise ValueError("GEMINI_API_KEY not set.")
if not GROQ_API_KEY:
    logging.warning("GROQ_API_KEY environment variable not set. Groq suggestions will be skipped.")
if not GOOGLE_CREDS_JSON:
    logging.error("GOOGLE_CREDS_JSON environment variable not set. Chat history will not be saved.")
    # In production, you might want to raise an exception here
    # raise ValueError("GOOGLE_CREDS_JSON not set.")


@app.post("/chat")
async def handle_chat(request: ChatRequest):
    session_id = request.session_id or f"session_{uuid.uuid4().hex}"
    effective_query = request.query
    
    format_as_table = contains_table_request(effective_query)
    logging.info(f"Session {session_id}: Query '{effective_query}' received. Table format requested: {format_as_table}")

    try:
        # Initialize components
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="Gemini API Key is not set. Please configure GEMINI_API_KEY.")
        
        llm = init_gemini_llm(GEMINI_API_KEY)
        embedding = init_embeddings(GEMINI_API_KEY) # Pass API key to init_embeddings as it uses Gemini
        db = load_chroma_db(embedding)
        retriever = db.as_retriever(search_kwargs={"k": 5})
        
        # Create conversational chain, passing Google Sheet credentials
        conversational_qa_chain = create_conversational_chain(llm, retriever, GOOGLE_CREDS_JSON)
        
        # Extract metadata from query
        spiritual_concept = extract_spiritual_concept(effective_query)
        life_problem = extract_life_problem(effective_query)
        scripture_source = extract_scripture_source(effective_query)

        # Get RAG result
        logging.info(f"Session {session_id}: Invoking RAG chain for query: '{effective_query}'")
        rag_result = await conversational_qa_chain.ainvoke({ # Use ainvoke for async
            "question": effective_query,
            # Note: conversational_qa_chain expects 'question' and 'chat_history' as inputs.
            # The other params (spiritual_concept, life_problem, scripture_source) are used within the RAG prompt itself
            # if defined in scripture_prompt.
            "spiritual_concept": spiritual_concept, # Passed if your scripture_prompt supports them
            "life_problem": life_problem,          # Passed if your scripture_prompt supports them
            "scripture_source": scripture_source   # Passed if your scripture_prompt supports them
        }, config={"configurable": {"session_id": session_id}})
        
        raw_rag_answer = rag_result.get("answer", "Could not retrieve from knowledge base.")
        rag_answer = clean_response_text(raw_rag_answer)
        logging.info(f"Session {session_id}: RAG raw answer length: {len(raw_rag_answer)}")
        
        # Get suggestions from other models
        suggestions = {"llama": "N/A", "mixtral": "N/A", "gemma": "N/A"}
        if GROQ_API_KEY:
            logging.info(f"Session {session_id}: Fetching Groq suggestions.")
            raw_suggestions = cached_groq_answers(
                effective_query,
                GROQ_API_KEY,
                spiritual_concept, # These parameters are for Groq prompt context
                life_problem,      # These parameters are for Groq prompt context
                scripture_source   # These parameters are for Groq prompt context
            )
            suggestions = clean_suggestions(raw_suggestions)
            logging.info(f"Session {session_id}: Groq suggestions fetched.")
        else:
            logging.warning(f"Session {session_id}: GROQ_API_KEY not set. Skipping Groq suggestions.")

        # Merge RAG answer and Groq suggestions (merge_groq_and_rag_answers expects the 'llm' instance)
        final_answer = await merge_groq_and_rag_answers(
            llm, # Pass the llm instance for the merging step
            rag_answer,
            suggestions,
            spiritual_concept,
            life_problem,
            scripture_source,
            format_as_table
        )
        final_answer = clean_response_text(final_answer)
        
        return {
            "answer": final_answer,
            "suggestions": suggestions,
            "session_id": session_id
        }
        
    except Exception as e:
        logging.error(f"API Error for session {session_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to ensure the API is running."""
    return {"status": "healthy", "message": "Hindu Scripture Advisor API is running"}

@app.get("/sessions/{session_id}/history")
async def get_chat_history_endpoint(session_id: str):
    """Get chat history for a session from Google Sheets."""
    try:
        if not GOOGLE_CREDS_JSON:
            raise HTTPException(status_code=500, detail="Google Credentials JSON (GOOGLE_CREDS_JSON) not set for history.")
        
        # Use get_session_history from qa_chain.py to retrieve the history manager
        history_manager = get_session_history(session_id, GOOGLE_CREDS_JSON)
        messages = []
        for message in history_manager.messages: # Access the messages property
            content = clean_response_text(message.content) if hasattr(message, 'content') else str(message)
            messages.append({
                "type": message.type,
                "content": content
            })
        logging.info(f"Retrieved {len(messages)} history entries for session {session_id}.")
        return {
            "session_id": session_id,
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        logging.error(f"Error retrieving history for session {session_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve chat history")

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear chat history for a session in Google Sheets."""
    try:
        if not GOOGLE_CREDS_JSON:
            raise HTTPException(status_code=500, detail="Google Credentials JSON (GOOGLE_CREDS_JSON) not set for history.")
        
        # Use get_session_history from qa_chain.py to retrieve the history manager
        history_manager = get_session_history(session_id, GOOGLE_CREDS_JSON)
        history_manager.clear() # Call the clear method on the history manager
        logging.info(f"Cleared session {session_id}")
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        logging.error(f"Error clearing session {session_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not clear session: {str(e)}")


# Configure logging (important for deployment visibility)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    import uvicorn
    # Render provides the PORT environment variable. Default to 8000 for local testing.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

