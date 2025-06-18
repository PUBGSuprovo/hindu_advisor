#fastapi_app..py


from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import logging
from config import GEMINI_API_KEY, GROQ_API_KEY
from llms import init_gemini_llm, init_embeddings
from vector_db import load_chroma_db
from qa_chain import create_conversational_chain, get_session_history
from groq_api import cached_groq_answers
from utils import extract_spiritual_concept, extract_life_problem, extract_scripture_source
import uuid
import re
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

def clean_response_text(text: str) -> str:
    """Clean up formatting issues in AI responses"""
    if not text:
        return text
    
    # Fix triple asterisks and other markdown formatting issues
    text = text.replace('***', '**')
    text = text.replace('****', '**')
    text = text.replace('*****', '**')
    
    # Fix broken bullet points that might appear as ***
    text = re.sub(r'^\*\*\*([^*])', r'• \1', text, flags=re.MULTILINE)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple line breaks to double
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Leading whitespace on lines
    
    # Fix common formatting issues
    text = re.sub(r'\*\*\s*\*\*', '**', text)  # ** ** to **
    text = re.sub(r'\*\*([^*]+)\*\*\*', r'**\1**', text)  # **text*** to **text**
    
    # Ensure proper sentence spacing
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    
    return text.strip()

def clean_suggestions(suggestions: dict) -> dict:
    """Clean up all suggestion texts"""
    cleaned = {}
    for key, value in suggestions.items():
        if isinstance(value, str):
            cleaned[key] = clean_response_text(value)
        else:
            cleaned[key] = value
    return cleaned

@app.post("/chat")
async def handle_chat(request: ChatRequest):
    session_id = request.session_id or f"session_{uuid.uuid4().hex}"
    effective_query = request.query

    try:
        # Initialize components
        llm = init_gemini_llm(GEMINI_API_KEY)
        embedding = init_embeddings()
        db = load_chroma_db(embedding)
        retriever = db.as_retriever(search_kwargs={"k": 5})
        
        # Create conversational chain
        conversational_qa_chain = create_conversational_chain(llm, retriever)
        
        # Get RAG result
        rag_result = conversational_qa_chain.invoke({
            "question": effective_query,
            "spiritual_concept": extract_spiritual_concept(effective_query),
            "life_problem": extract_life_problem(effective_query),
            "scripture_source": extract_scripture_source(effective_query)
        }, config={"configurable": {"session_id": session_id}})
        
        # Extract and clean the answer
        raw_answer = rag_result.get("answer", "Could not retrieve from knowledge base.")
        rag_answer = clean_response_text(raw_answer)
        
        # Get suggestions from other models
        suggestions = {"llama": "N/A", "mixtral": "N/A", "gemma": "N/A"}
        if GROQ_API_KEY:
            raw_suggestions = cached_groq_answers(
                effective_query,
                GROQ_API_KEY,
                extract_spiritual_concept(effective_query),
                extract_life_problem(effective_query),
                extract_scripture_source(effective_query)
            )
            suggestions = clean_suggestions(raw_suggestions)
        
        # Log the cleaning for debugging
        if raw_answer != rag_answer:
            logging.info(f"Cleaned response for session {session_id}")
        
        return {
            "answer": rag_answer,
            "suggestions": suggestions,
            "session_id": session_id
        }
        
    except Exception as e:
        logging.error(f"API Error for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Hindu Scripture Advisor API is running"}

@app.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        history = get_session_history(session_id)
        messages = []
        for message in history.messages:
            messages.append({
                "type": message.type,
                "content": clean_response_text(message.content) if hasattr(message, 'content') else str(message)
            })
        return {
            "session_id": session_id,
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        logging.error(f"Error retrieving history for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not retrieve chat history")

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear chat history for a session"""
    try:
        if session_id in store:
            del store[session_id]
            logging.info(f"Cleared session {session_id}")
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            return {"message": f"Session {session_id} not found"}
    except Exception as e:
        logging.error(f"Error clearing session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not clear session")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)