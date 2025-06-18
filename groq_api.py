# groq_api.py
import requests
import logging
from concurrent.futures import ThreadPoolExecutor

def groq_scripture_answer(
    model_name: str,
    query: str,
    groq_api_key: str,
    spiritual_concept: str = "general",
    life_problem: str = "guidance",
    scripture_source: str = "Hindu scriptures"
) -> str:
    """
    Fetches a concise spiritual guidance answer from a specific Groq model.

    Args:
        model_name (str): The name of the Groq model to use (e.g., "llama", "mixtral", "gemma").
        query (str): The user's original query.
        groq_api_key (str): Your Groq API key.
        spiritual_concept (str): Extracted spiritual concept from the query.
        life_problem (str): Extracted life problem from the query.
        scripture_source (str): Extracted scripture source preference from the query.

    Returns:
        str: The concise guidance from the Groq model, or an error message.
    """
    if not groq_api_key:
        return "Groq API key not available. Skipping suggestions."
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        # Mapping simplified model names to actual Groq API model identifiers
        groq_model_map = {
            "llama": "llama3-70b-8192",
            "mixtral": "mixtral-8x7b-32976",
            "gemma": "gemma2-9b-it"
        }
        actual_model_name = groq_model_map.get(model_name.lower(), model_name)

        # Crafting a concise prompt for Groq models
        prompt_content = (
            f"User query: '{query}'. "
            f"Provide a concise, practical spiritual guidance or answer related to "
            f"**{spiritual_concept}** for **{life_problem}**, referencing "
            f"**{scripture_source}** if applicable. Be brief and to the point."
        )
        payload = {
            "model": actual_model_name,
            "messages": [{"role": "user", "content": prompt_content}],
            "temperature": 0.5, # Balanced creativity
            "max_tokens": 250   # Keep responses concise
        }
        logging.info(f"Calling Groq API: {actual_model_name} for query: '{query}'")
        response = requests.post(url, headers=headers, json=payload, timeout=15) # Reduced timeout slightly
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        return data['choices'][0]['message']['content'] if 'choices' in data and data['choices'] else f"Empty response from {model_name}."
    except requests.exceptions.Timeout:
        logging.warning(f"Timeout error from Groq model {model_name} for query: '{query}'")
        return f"Timeout error from {model_name}."
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error from Groq model {model_name} for query '{query}': {e}")
        return f"Request error from {model_name}: {e}"
    except Exception as e:
        logging.error(f"Unexpected error from Groq model {model_name} for query '{query}': {e}")
        return f"Error from {model_name}: {e}"

def cached_groq_answers(
    query: str,
    groq_api_key: str,
    spiritual_concept: str,
    life_problem: str,
    scripture_source: str
) -> dict:
    """
    Fetches and caches Groq answers from multiple models concurrently.

    Args:
        query (str): The user's current query.
        groq_api_key (str): Your Groq API key.
        spiritual_concept (str): Extracted spiritual concept.
        life_problem (str): Extracted life problem.
        scripture_source (str): Extracted scripture source.

    Returns:
        dict: A dictionary where keys are model names and values are their responses.
    """
    logging.info(f"Fetching cached Groq answers for query: '{query}'")
    models = ["llama", "mixtral", "gemma"]
    results = {}

    if not groq_api_key:
        logging.warning("Groq API key not provided. Skipping all Groq suggestions.")
        return {k: "Groq API key not available." for k in models}

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        # Submit tasks to the thread pool, passing the API key
        future_to_model = {
            executor.submit(groq_scripture_answer, name, query, groq_api_key, spiritual_concept, life_problem, scripture_source): name
            for name in models
        }
        for future in future_to_model:
            model_name = future_to_model[future]
            try:
                results[model_name] = future.result()
            except Exception as e:
                logging.error(f"Error fetching result for Groq model {model_name}: {e}")
                results[model_name] = f"Failed to get answer: {e}"
    return results
