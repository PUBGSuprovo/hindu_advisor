# utils.py
import string
import logging
import re # Ensure re is imported for clean_response_text
from typing import List, Dict # Import Dict for type hinting
from langchain.prompts import PromptTemplate # Ensure PromptTemplate is imported
from langchain_core.language_models import BaseChatModel # Import BaseChatModel for type hinting

# Define lists of keywords for categorization and intent detection
GREETINGS: List[str] = ["hi", "hello", "hey", "namaste", "yo", "pranam", "jai shree ram", "om namah shivaya", "radhe radhe", "good morning", "good afternoon", "good evening"]
TASK_KEYWORDS: List[str] = [
    "dharma", "karma", "moksha", "atman", "brahman", "yoga", "meditation", "bhakti", "jnana", "seva",
    "stress", "anxiety", "fear", "sadness", "anger", "grief", "purpose", "meaning of life", "suffering",
    "bhagavad gita", "veda", "upanishad", "purana", "ramayana", "mahabharata", "scripture", "text", "shastra",
    "guidance", "solution", "advice", "teachings", "principles", "philosophy", "answer", "explain", "meaning",
    "table", "format", "chart", "show", "give", "list", "bullet", "points", "itemize", "enumerate"
]
FORMATTING_KEYWORDS: List[str] = ["table", "tabular", "chart", "format", "list", "bullet", "points", "itemize", "enumerate"]


def _clean_query(query: str) -> str:
    """Helper function to clean and normalize a query string."""
    return query.translate(str.maketrans('', '', string.punctuation)).lower().strip()


def extract_spiritual_concept(query: str) -> str:
    """
    Extracts a primary spiritual concept from the query.
    Returns "general" if no specific concept is identified.
    """
    q = _clean_query(query)
    concepts = {
        "dharma": ["dharma", "duty", "righteousness"],
        "karma": ["karma", "action", "consequences"],
        "moksha": ["moksha", "liberation", "salvation"],
        "atman": ["atman", "soul", "self"],
        "brahman": ["brahman", "ultimate reality", "god"],
        "yoga": ["yoga", "meditation", "mindfulness", "union"],
        "bhakti": ["bhakti", "devotion", "love"],
        "jnana": ["jnana", "knowledge", "wisdom"],
        "seva": ["seva", "service", "selfless service"]
    }
    for concept, kws in concepts.items():
        if any(k in q for k in kws):
            logging.info(f"Detected spiritual concept: {concept}")
            return concept
    logging.info("No specific spiritual concept detected, defaulting to 'general'.")
    return "general"

def extract_life_problem(query: str) -> str:
    """
    Extracts a common life problem from the query.
    Returns "guidance" if no specific problem is identified.
    """
    q = _clean_query(query)
    problems = {
        "stress": ["stress", "stressed", "anxiety", "anxious", "tension"],
        "fear": ["fear", "scared", "afraid", "fright"],
        "sadness": ["sadness", "sad", "depressed", "unhappy", "sorrow"],
        "anger": ["anger", "angry", "frustration", "irritation"],
        "grief": ["grief", "loss", "mourning"],
        "purpose": ["purpose", "meaning of life", "direction"],
        "suffering": ["suffering", "pain", "hardship"],
        "relationship": ["relationship", "relationships", "family", "friends", "love", "marriage", "partner"],
        "decision": ["decision", "choice", "dilemma"]
    }
    for problem, kws in problems.items():
        if any(k in q for k in kws):
            logging.info(f"Detected life problem: {problem}")
            return problem
    logging.info("No specific life problem detected, defaulting to 'guidance'.")
    return "guidance"

def extract_scripture_source(query: str) -> str:
    """
    Extracts a specific Hindu scripture source from the query if mentioned.
    Returns "Hindu scriptures" if no specific source is identified.
    """
    q = query.lower()
    sources = {
        "bhagavad gita": ["bhagavad gita", "gita"],
        "veda": ["veda", "vedas", "rigveda", "yajurveda", "samaveda", "atharvaveda"],
        "upanishad": ["upanishad", "upanishads"],
        "purana": ["purana", "puranas", "vishnu purana", "bhagavata purana", "garuda purana"],
        "ramayana": ["ramayana", "ramayan"],
        "mahabharata": ["mahabharata", "mahabharat"],
        "yoga sutras": ["yoga sutras", "patanjali"],
        "dharma shastras": ["dharma shastras", "manu smriti"],
        "hatha yoga pradipika": ["hatha yoga pradipika"],
        "shiva sutras": ["shiva sutras"]
    }
    for source, kws in sources.items():
        if any(k in q for k in kws):
            formatted_source = " ".join([w.capitalize() for w in source.split()])
            logging.info(f"Detected scripture source: {formatted_source}")
            return formatted_source
    logging.info("No specific scripture source detected, defaulting to 'Hindu scriptures'.")
    return "Hindu scriptures"


def clean_response_text(text: str) -> str:
    """Clean up formatting issues in AI responses"""
    if not text:
        return text
    
    # Fix triple asterisks and other markdown formatting issues
    text = text.replace('***', '**')
    text = text.replace('****', '**')
    text = text.replace('*****', '**')
    
    # Fix broken bullet points that might appear as ***
    text = re.sub(r'^\*\s*\*', '*', text, flags=re.MULTILINE)
    text = re.sub(r'\n\*\s*\*', '\n*', text) # Newlines followed by **
    
    # Remove leading/trailing spaces on each line
    text = "\n".join([line.strip() for line in text.split('\n')])
    
    return text

def clean_suggestions(suggestions: Dict[str, str]) -> Dict[str, str]:
    """Clean up formatting in suggestions from Groq models."""
    cleaned = {}
    for model, text in suggestions.items():
        cleaned[model] = clean_response_text(text)
    return cleaned

def contains_table_request(query: str) -> bool:
    """
    Checks if the user's query explicitly asks for a table format.
    """
    q_lower = query.lower()
    return any(keyword in q_lower for keyword in FORMATTING_KEYWORDS)

# --- Merge Prompt Templates ---

merge_prompt_template_default = """
You are a Hindu scripture and spiritual guidance assistant.
Your goal is to synthesize information from a primary RAG-based answer and several other AI suggestions into a single, coherent, and practical guidance or answer for **{life_problem}**, referencing **{spiritual_concept}** from **{scripture_source}** if specified.
Prioritize the Primary RAG Answer. If it's weak or irrelevant, use Additional Suggestions.
Ensure the final guidance is clear, actionable, and respectful of Hindu traditions.
Present the final guidance as a clear, actionable paragraph or list.

If the user's input was *only* a greeting, respond politely. For inputs that include a greeting but also contain a query, focus on answering the query.

Primary RAG Answer:
{rag}

Additional Suggestions:
- LLaMA Suggestion: {llama}
- Mixtral Suggestion: {mixtral}
- Gemma Suggestion: {gemma}

Refined and Merged Guidance (Tailored for {spiritual_concept}, {life_problem}, from {scripture_source}):
"""

merge_prompt_default = PromptTemplate.from_template(merge_prompt_template_default)


merge_prompt_template_table = """
You are a Hindu scripture and spiritual guidance assistant.
Your goal is to synthesize information from a primary RAG-based answer and several other AI suggestions into a single, coherent, and practical guidance or answer for **{life_problem}**, referencing **{spiritual_concept}** from **{scripture_source}** if specified.
Prioritize the Primary RAG Answer. If it's weak or irrelevant, use Additional Suggestions.
Ensure the final guidance is clear, actionable, and respectful of Hindu traditions.

**You MUST present the final guidance as a clear markdown table if appropriate. Include columns for Concept/Teaching, Scripture Reference, and Practical Application.**

If the user's input was *only* a greeting, respond politely. For inputs that include a greeting but also contain a query, focus on answering the query.

Primary RAG Answer:
{rag}

Additional Suggestions:
- LLaMA Suggestion: {llama}
- Mixtral Suggestion: {mixtral}
- Gemma Suggestion: {gemma}

Refined and Merged Guidance (Tailored for {spiritual_concept}, {life_problem}, from {scripture_source}, in markdown table format if applicable):
"""

merge_prompt_table = PromptTemplate.from_template(merge_prompt_template_table)

async def merge_groq_and_rag_answers(
    llm: BaseChatModel,
    rag_answer: str,
    groq_suggestions: Dict[str, str],
    spiritual_concept: str,
    life_problem: str,
    scripture_source: str,
    format_as_table: bool = False
) -> str:
    """
    Merges RAG answer and Groq suggestions using a chosen LLM based on user's query intent.
    """
    logging.info(f"Merging answers with table format: {format_as_table}")
    
    prompt_template = merge_prompt_table if format_as_table else merge_prompt_default

    chain = prompt_template | llm

    response = await chain.ainvoke({
        "rag": rag_answer,
        "llama": groq_suggestions.get("llama", "N/A"),
        "mixtral": groq_suggestions.get("mixtral", "N/A"),
        "gemma": groq_suggestions.get("gemma", "N/A"),
        "spiritual_concept": spiritual_concept,
        "life_problem": life_problem,
        "scripture_source": scripture_source
    })
    
    # Direct access to content if it's a message object, otherwise assume string
    return response.content if hasattr(response, 'content') else str(response)
