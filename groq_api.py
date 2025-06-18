# groq_api.py
import requests
import logging
from concurrent.futures import ThreadPoolExecutor

# GROQ_API_KEY is now expected to be passed as an argument to the functions
def groq_scripture_answer(
    model_name: str,
    query: str,
    groq_api_key: str, # ADDED API KEY AS A PARAMETER
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
        
        # Craft a prompt specifically for Groq models to give concise suggestions
        # Tailored to the extracted spiritual_concept, life_problem, and scripture_source
        system_message = (
            f"You are a highly concise spiritual advisor focused on {scripture_source}. "
            f"Provide a brief, practical insight related to '{spiritual_concept}' and '{life_problem}'."
            f"Keep your response under 50 words and avoid conversational filler."
        )
        
        payload = {
            "model": f"{model_name}-8b-it" if model_name == "llama" else model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ],
            "temperature": 0.3, # Keep temperature low for concise, direct answers
            "max_tokens": 100 # Adjust max_tokens to control response length
        }

        response = requests.post(url, headers=headers, json=payload, timeout=10) # Added timeout
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        logging.info(f"Fetched Groq answer from {model_name}.")
        return answer
    except requests.exceptions.Timeout:
        logging.error(f"Groq API call to {model_name} timed out for query: '{query}'.")
        return f"Groq ({model_name}) timed out."
    except requests.exceptions.RequestException as e:
        logging.error(f"Groq API call to {model_name} failed: {e} for query: '{query}'.", exc_info=True)
        return f"Groq ({model_name}) error: {e}"
    except Exception as e:
        logging.error(f"Unexpected error with Groq API for {model_name}: {e}", exc_info=True)
        return f"Groq ({model_name}) unexpected error."

def cached_groq_answers(
    query: str,
    groq_api_key: str,
    spiritual_concept: str = "general",
    life_problem: str = "guidance",
    scripture_source: str = "Hindu scriptures"
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
                logging.error(f"Error fetching Groq answer for {model_name}: {e}", exc_info=True)
                results[model_name] = f"Error: {e}"
    logging.info("Completed fetching Groq answers.")
    return results

