# utils.py
import string
import logging
from typing import List

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
    return query.translate(str.maketrans('', '', string.punctuation)).strip().lower()

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
    # Check if it's a known greeting and not too long, and doesn't contain task keywords
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

    # Must contain at least one formatting keyword
    if not any(keyword in cleaned_query for keyword in FORMATTING_KEYWORDS):
        return False

    # Remove common filler words and formatting keywords to see if any substantive words remain
    filler_words = ["in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you", "provide", "the", "an", "this", "my", "your", "for", "me"]
    non_formatting_or_filler_words = [
        w for w in words if w not in FORMATTING_KEYWORDS and w not in filler_words
    ]

    # If there are very few substantive words left (e.g., 0 or 1), it's likely a formatting request
    return len(non_formatting_or_filler_words) <= 1


def extract_spiritual_concept(query: str) -> str:
    """
    Extracts a spiritual concept from the query if mentioned.
    Returns "general" if no specific concept is identified.
    """
    q = query.lower()
    concepts = {
        "dharma": ["dharma", "duty", "righteousness"],
        "karma": ["karma", "action", "consequence", "karma yoga"],
        "moksha": ["moksha", "liberation", "salvation", "enlightenment"],
        "atman": ["atman", "soul", "self"],
        "brahman": ["brahman", "ultimate reality", "absolute truth"],
        "yoga": ["yoga", "meditation", "union", "asanas"],
        "bhakti": ["bhakti", "devotion", "bhakti yoga"],
        "jnana": ["jnana", "knowledge", "wisdom", "jnana yoga"],
        "seva": ["seva", "selfless service"],
        "reincarnation": ["reincarnation", "rebirth", "samsara"],
        "maya": ["maya", "illusion", "worldly illusion"],
        "nirvana": ["nirvana", "spiritual liberation"]
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
    q = query.lower()
    problems = {
        "stress": ["stress", "tension", "anxiety", "worry", "overwhelmed"],
        "anger": ["anger", "frustration", "irritation", "rage"],
        "grief": ["grief", "loss", "sadness", "sorrow", "bereavement"],
        "purpose": ["purpose", "meaning of life", "direction", "aim", "goal", "why am i here"],
        "fear": ["fear", "insecurity", "doubt", "apprehension", "courage"],
        "relationships": ["relationship", "family", "friends", "love", "conflict", "breakup", "marriage"],
        "suffering": ["suffering", "pain", "hardship", "adversity", "misery"],
        "decision making": ["decision", "choice", "dilemma", "confused", "uncertainty"]
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


def contains_table_request(query: str) -> bool:
    """
    Checks if the user's query explicitly asks for a table format.
    """
    q = query.lower()
    return any(k in q for k in ["table", "tabular", "chart", "in a table", "in table format", "as a table"])