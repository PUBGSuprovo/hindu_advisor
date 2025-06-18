# utils.py
import string
import logging
import re
from typing import List, Dict # Import Dict for type hinting
from langchain.prompts import PromptTemplate # Ensure PromptTemplate is imported
from langchain_core.language_models import BaseChatModel # Import BaseChatModel for type hinting
from langchain.chains.llm import LLMChain # Import LLMChain for the merge function

# Define lists of keywords for categorization and intent detection
GREETINGS: List[str] = [
    "hi", "hello", "hey", "namaste", "yo", "pranam", "jai shree ram", "om namah shivaya",
    "radhe radhe", "good morning", "good afternoon", "good evening"
]
TASK_KEYWORDS: List[str] = [
    "dharma", "karma", "moksha", "atman", "brahman", "yoga", "meditation", "bhakti", "jnana", "seva",
    "stress", "anxiety", "fear", "sadness", "anger", "grief", "purpose", "meaning of life", "suffering",
    "bhagavad gita", "veda", "upanishad", "purana", "ramayana", "mahabharata", "scripture", "text", "shastra",
    "guidance", "solution", "advice", "teachings", "principles", "philosophy", "answer", "explain", "meaning",
    "table", "format", "chart", "show", "give", "list", "bullet", "points", "itemize", "enumerate"
]
FORMATTING_KEYWORDS: List[str] = ["table", "tabular", "chart", "format", "list", "bullet", "points", "itemize", "enumerate", "in a table", "as a table"]


def _clean_query(query: str) -> str:
    """Helper function to clean and normalize a query string."""
    return query.translate(str.maketrans('', '', string.punctuation)).lower().strip()

def is_greeting(query: str) -> bool:
    """
    Checks if a query is purely a greeting.

    A greeting is considered a short phrase from the GREETINGS list,
    without containing any specific task-related keywords.
    """
    if not query:
        return False
    cleaned_query = _clean_query(query)
    words = cleaned_query.split()
    is_pure_greeting = cleaned_query in GREETINGS and len(words) <= 3
    contains_task_keywords = any(k in cleaned_query for k in TASK_KEYWORDS)
    return is_pure_greeting and not contains_task_keywords


def is_formatting_request(query: str) -> bool:
    """
    Checks if a query is primarily a request for formatting (e.g., "show in table").

    It identifies queries that contain formatting keywords and very few other
    substantive words.
    """
    if not query:
        return False
    cleaned_query = _clean_query(query)
    words = cleaned_query.split()

    if not any(keyword in cleaned_query for keyword in FORMATTING_KEYWORDS):
        return False

    filler_words = ["in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you", "provide", "the", "an", "this", "my", "your", "for"]
    non_formatting_or_filler_words = [
        w for w in words if w not in FORMATTING_KEYWORDS and w not in filler_words
    ]

    return len(non_formatting_or_filler_words) <= 1


def extract_spiritual_concept(query: str) -> str:
    """
    Extracts a spiritual concept from the query if mentioned.
    Returns "general" if no specific concept is identified.
    """
    q = _clean_query(query)
    concepts = {
        "dharma": ["dharma", "duty", "righteousness"],
        "karma": ["karma", "action", "consequences"],
        "moksha": ["moksha", "liberation", "salvation", "enlightenment"],
        "atman": ["atman", "soul", "self"],
        "brahman": ["brahman", "ultimate reality", "absolute truth"],
        "yoga": ["yoga", "meditation", "union", "asanas"],
        "bhakti": ["bhakti", "devotion", "bhakti yoga"],
        "jnana": ["jnana", "knowledge", "wisdom", "jnana yoga"],
        "seva": ["seva", "selfless service"],
        "reincarnation": ["reincarnation", "rebirth", "samsara"],
        "maya": ["maya", "illusion", "worldly illusion"],
        "nirvana": ["nirvana", "spiritual liberation"],
        "guna": ["guna", "qualities", "modes of nature"],
        "sanskara": ["sanskara", "impressions", "mental imprints"],
        "sattva": ["sattva", "purity", "goodness"],
        "rajas": ["rajas", "passion", "activity"],
        "tamas": ["tamas", "ignorance", "darkness"]
    }
    for concept, kws in concepts.items():
        if any(k in q for k in kws):
            logging.info(f"Detected spiritual concept: {concept}")
            return concept
    logging.info("No specific spiritual concept detected, defaulting to 'general'.")
    return "general"


def extract_life_problem(query: str) -> str:
    """
    Extracts a common life problem from the query if mentioned.
    Returns "guidance" if no specific problem is identified.
    """
    q = _clean_query(query)
    problems = {
        "stress": ["stress", "tension", "anxiety", "worry", "overwhelmed", "pressure"],
        "anger": ["anger", "frustration", "irritation", "rage", "resentment"],
        "grief": ["grief", "loss", "sadness", "sorrow", "bereavement", "heartbreak"],
        "purpose": ["purpose", "meaning of life", "direction", "aim", "goal", "why am i here", "lack of direction"],
        "fear": ["fear", "insecurity", "doubt", "apprehension", "courage", "hesitation"],
        "relationships": ["relationship", "family", "friends", "love", "conflict", "breakup", "marriage", "loneliness", "social issues"],
        "suffering": ["suffering", "pain", "hardship", "adversity", "misery", "struggle"],
        "decision making": ["decision", "choice", "dilemma", "confused", "uncertainty", "indecision"],
        "materialism": ["materialism", "attachment", "desire", "greed"],
        "ego": ["ego", "pride", "self-importance", "arrogance"],
        "depression": ["depression", "despair", "hopelessness", "melancholy"]
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
        "bhagavad gita": ["bhagavad gita", "gita", "bhagwad geeta"],
        "veda": ["veda", "vedas", "rigveda", "yajurveda", "samaveda", "atharvaveda"],
        "upanishad": ["upanishad", "upanishads"],
        "purana": ["purana", "puranas", "vishnu purana", "bhagavata purana", "garuda purana", "skanda purana"],
        "ramayana": ["ramayana", "ramayan", "valmiki ramayana"],
        "mahabharata": ["mahabharata", "mahabharat"],
        "yoga sutras": ["yoga sutras", "patanjali yoga sutras", "patanjali"],
        "dharma shastras": ["dharma shastras", "manu smriti"],
        "hatha yoga pradipika": ["hatha yoga pradipika"],
        "shiva sutras": ["shiva sutras"],
        "brahma sutras": ["brahma sutras"],
        "vedanta": ["vedanta"]
    }
    for source, kws in sources.items():
        if any(k in q for k in kws):
            formatted_source = " ".join([w.capitalize() for w in source.split()])
            logging.info(f"Detected scripture source: {formatted_source}")
            return formatted_source
    logging.info("No specific scripture source detected, defaulting to 'Hindu scriptures'.")
    return "Hindu scriptures"


def contains_table_request(query: str) -> bool:
    """
    Checks if the user's query explicitly asks for a table format.
    """
    q_lower = query.lower()
    return any(k in q_lower for k in FORMATTING_KEYWORDS)


def clean_response_text(text: str) -> str:
    """Clean up formatting issues in AI responses."""
    if not text:
        return text
    
    # Replace triple/quadruple/quintuple asterisks with double asterisks for bold
    text = text.replace('***', '**').replace('****', '**').replace('*****', '**')
    
    # Fix broken bullet points (e.g., *** or ** followed by text)
    # This regex looks for ** followed by an optional space, then any character not a space or newline, at the start of a line.
    # Replace it with '- ' for consistent markdown bullets.
    text = re.sub(r'^\*\*\s*([^\s\n])', r'- \1', text, flags=re.MULTILINE)
    # Also handle single asterisks at the start of a line that might be intended as bullets
    text = re.sub(r'^\*\s*([^\s\n])', r'- \1', text, flags=re.MULTILINE)
    
    # Clean up excessive whitespace (multiple line breaks to double, multiple spaces/tabs to single)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text) # Multiple line breaks to double
    text = re.sub(r'[ \t]+', ' ', text)          # Multiple spaces/tabs to single space
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE) # Leading whitespace on lines
    
    # Ensure proper sentence spacing after periods, if not already present
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    
    # Remove leading/trailing spaces on each line for cleaner output
    text = "\n".join([line.strip() for line in text.split('\n')])
    
    return text.strip()


def clean_suggestions(suggestions: Dict[str, str]) -> Dict[str, str]:
    """Clean up formatting in suggestions from Groq models."""
    cleaned = {}
    for model, text in suggestions.items():
        cleaned[model] = clean_response_text(text)
    return cleaned

# --- Merge Prompt Templates (Moved from prompts.py for direct import) ---
# Ensure these are the same as in your prompts.py file

merge_prompt_template_default = """
You are a Hindu scripture and spiritual guidance assistant.
Your goal is to synthesize information from a primary RAG-based answer and several other AI suggestions into a single, coherent, and practical guidance or answer for **{life_problem}**, referencing **{spiritual_concept}** from **{scripture_source}** if specified.
Prioritize the Primary RAG Answer. If it's weak or irrelevant, use Additional Suggestions to enrich or provide alternatives.
Ensure the final guidance is clear, actionable, and respectful of Hindu traditions. Present as a clear paragraph or a list of points.
If the user's input was *only* a greeting, respond politely (e.g., "Namaste! How can I assist you with Hindu scriptures today?"). For inputs that include a greeting but also contain a query, focus on answering the query.

Primary RAG Answer:
{rag}

Additional Suggestions (consider these, but prioritize the RAG answer):
- LLaMA Suggestion: {llama}
- Mixtral Suggestion: {mixtral}
- Gemma Suggestion: {gemma}

Refined and Merged Guidance (Tailored for {spiritual_concept}, {life_problem}, from {scripture_source}):
"""
merge_prompt_default = PromptTemplate.from_template(merge_prompt_template_default)


merge_prompt_template_table = """
You are a Hindu scripture and spiritual guidance assistant.
Your goal is to synthesize information from a primary RAG-based answer and several other AI suggestions into a single, coherent, and practical guidance or answer for **{life_problem}**, referencing **{spiritual_concept}** from **{scripture_source}** if specified.
Prioritize the Primary RAG Answer. If it's weak or irrelevant, use Additional Suggestions to enrich or provide alternatives.
Ensure the final guidance is clear, actionable, and respectful of Hindu traditions.

**You MUST present the final guidance as a clear markdown table if appropriate. Include columns for 'Spiritual Concept/Teaching', 'Scripture Reference/Source', and 'Practical Application/Guidance'. If a direct scripture reference isn't available, mention 'General Hindu Wisdom'.**
If the information doesn't naturally fit into a table, clearly state why and present it in a well-structured paragraph format.

If the user's input was *only* a greeting, respond politely (e.g., "Namaste! How can I assist you with Hindu scriptures today?"). For inputs that include a greeting but also contain a query, focus on answering the query.

Primary RAG Answer:
{rag}

Additional Suggestions (consider these, but prioritize the RAG answer):
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

    # Create an LLMChain directly using the selected prompt template and LLM
    merge_chain = LLMChain(llm=llm, prompt=prompt_template)

    try:
        # Invoke the chain with all necessary variables for the prompt
        merged_result_obj = await merge_chain.ainvoke({
            "rag": rag_answer,
            "llama": groq_suggestions.get("llama", "N/A"),
            "mixtral": groq_suggestions.get("mixtral", "N/A"),
            "gemma": groq_suggestions.get("gemma", "N/A"),
            "spiritual_concept": spiritual_concept,
            "life_problem": life_problem,
            "scripture_source": scripture_source
        })
        
        # Extract content from the LLM's response object (which might be an AIMessage or a simple string)
        return merged_result_obj.content if hasattr(merged_result_obj, 'content') else str(merged_result_obj)
    except Exception as e:
        logging.error(f"Error during answer merging: {e}", exc_info=True)
        return f"Error merging answers: {e}. Here is the RAG answer: {rag_answer}"

