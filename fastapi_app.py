# Please paste your FastAPI code here.


import os
import json
import logging
import zipfile
import requests
import string
import re
import base64
import asyncio
import httpx # MODIFICATION: Using httpx for async requests
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any, Union

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Langchain and Google Generative AI imports
from langchain_chroma import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory # For in-memory
# MODIFICATION: Example import for robust session management
# from langchain_community.chat_message_histories import RedisChatMessageHistory 
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda 
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import StringPromptValue 


# Google Sheets logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Consolidated: Custom Callback for LangChain ---
class SafeTracer(BaseCallbackHandler):
    """
    A custom LangChain callback handler to safely log chain outputs.
    Avoids breaking if output structure is unexpected.
    Logs to Python's logging system.
    """
    def on_chain_end(self, outputs: Any, **kwargs): 
        try:
            if isinstance(outputs, (AIMessage, HumanMessage, SystemMessage)):
                logging.info(f"🔁 Chain ended. Message type: {type(outputs).__name__}, content snippet: {outputs.content[:100]}...")
            elif isinstance(outputs, StringPromptValue): 
                logging.info(f"🔁 Chain ended. PromptValue content snippet: {outputs.text[:100]}...")
            elif isinstance(outputs, dict):
                # MODIFICATION: Handle AgentAction Pydantic model logging
                if 'thought' in outputs:
                     logging.info(f"🔁 Chain ended. AgentAction thought: {outputs['thought']}")
                elif "answer" in outputs:
                    logging.info(f"🔁 Chain ended. Answer snippet: {outputs['answer'][:100]}...")
                elif "output" in outputs:
                    logging.info(f"🔁 Chain ended. Output snippet: {outputs['output'][:100]}...")
                elif "text" in outputs: 
                    logging.info(f"🔁 Chain ended. Text output snippet: {outputs['text'][:100]}...")
                else: 
                    logging.info(f"🔁 Chain ended. Output (type: {type(outputs)}, content snippet): {str(outputs)[:100]}...")
            elif isinstance(outputs, str): 
                logging.info(f"🔁 Chain ended. String output snippet: {outputs[:100]}...")
            else:
                # Catch-all for any other unexpected types
                logging.info(f"🔁 Chain ended. Output (type: {type(outputs)}, content snippet): {str(outputs)[:100]}...")
        except Exception as e:
            logging.error(f"❌ Error in on_chain_end callback: {e}")


# --- Consolidated: Vector Database Setup & Download ---
def download_and_extract_db_for_app():
    """
    Downloads and extracts a prebuilt ChromaDB from a HuggingFace URL.
    This function checks if the DB already exists to avoid re-downloading on restarts.
    """
    url = "https://huggingface.co/datasets/Dyno1307/chromadb-diet/resolve/main/db.zip"
    zip_path = "/tmp/db.zip" # Using /tmp for temporary storage on Render
    extract_path = "/tmp/chroma_db" # MUST match setup_vector_database's persist_directory

    os.makedirs(extract_path, exist_ok=True)

    if os.path.exists(os.path.join(extract_path, "index")): # Check for 'index' file within the extracted path
        logging.info("✅ Chroma DB already exists, skipping download.")
        return

    try:
        logging.info("⬇️ Downloading Chroma DB zip from HuggingFace...")
        # MODIFICATION: Using standard requests here is fine as it's a startup task, not in the request path.
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        logging.info(f"📦 Extracting zip to {extract_path}...")
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

def setup_vector_database(chroma_db_directory: str = "/tmp/chroma_db", in_memory: bool = False):
    """
    Initializes Chroma vector database using Gemini embeddings.
    """
    try:
        logging.info("🔧 Initializing Gemini Embeddings...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in environment variables. Please set it for Gemini API access.")
        
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        logging.info("✅ Gemini Embeddings loaded.")

        persist_path = None if in_memory else chroma_db_directory

        db = Chroma(
            persist_directory=persist_path,
            embedding_function=embedding
        )

        try:
            count = len(db.get()['documents'])
            logging.info(f"📦 Vector DB loaded with {count} documents.")
        except Exception as e:
            logging.warning(f"⚠️ Could not count documents in Vector DB (might be empty or first run): {e}")

        logging.info("✅ Chroma DB initialized successfully.")
        return db, embedding

    except Exception as e:
        logging.exception("❌ Vector DB setup failed.")
        raise

# --- MODIFICATION: Asynchronous Groq Integration ---
async def _groq_diet_answer_single_async(client: httpx.AsyncClient, model_name: str, groq_api_key: str, prompt_content: str) -> str:
    """Helper function to call a single Groq model asynchronously with retries."""
    groq_model_map = {
        "llama": "llama3-70b-8192",
        "gemma": "gemma2-9b-it",
        "mistral-saba": "mistral-saba-24b"
    }
    actual_model_name = groq_model_map.get(model_name.lower(), model_name)
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": actual_model_name,
        "messages": [{"role": "user", "content": prompt_content}],
        "temperature": 0.5,
        "max_tokens": 250
    }
    
    # MODIFICATION: Simple retry logic
    for attempt in range(2): # Retry once
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data and data.get('choices') and data['choices'][0].get('message'):
                return data['choices'][0]['message']['content']
            return f"No suggestion from {actual_model_name} (empty/malformed response)."
        except (httpx.TimeoutException, httpx.RequestError) as e:
            logging.warning(f"⚠️ Groq request failed for {model_name} on attempt {attempt+1}: {e}")
            if attempt == 1: # If last attempt failed
                return f"Error: Request to {model_name} failed after retries."
            await asyncio.sleep(1) # Wait before retrying
        except Exception as e:
            logging.error(f"❌ Unexpected error from {model_name}: {e}")
            return f"Error: Unexpected issue with {model_name}."
    return f"Error: Failed to get a response from {model_name}."


async def cached_groq_answers_async(query: str, groq_api_key: str, dietary_type: str, goal: str, region: str) -> dict:
    """Fetches diet suggestions from multiple Groq models in parallel asynchronously."""
    logging.info(f"Fetching Groq answers async for query: '{query}', pref: '{dietary_type}', goal: '{goal}', region: '{region}'")
    models = ["llama", "gemma", "mistral-saba"]
    if not groq_api_key:
        logging.warning("GROQ_API_KEY not available. Skipping Groq calls.")
        return {k: "Groq API key not available." for k in models}

    prompt_content = (
        f"User query: '{query}'. "
        f"Provide a concise, practical {dietary_type} diet suggestion or food item "
        f"for {goal}, tailored for a {region} Indian context. "
        f"Focus on readily available ingredients. Be brief and to the point."
    )

    async with httpx.AsyncClient() as client:
        tasks = [
            _groq_diet_answer_single_async(client, name, groq_api_key, prompt_content)
            for name in models
        ]
        results_list = await asyncio.gather(*tasks)

    return dict(zip(models, results_list))


# --- Consolidated: LangChain Chain Definitions ---
llm_chains_session_store = {} 

def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Retrieves or creates a LangChain ChatMessageHistory for a given session ID.
    MODIFICATION: Includes commented-out example for a persistent Redis store.
    """
    # --- In-Memory Store (Default) ---
    if session_id not in llm_chains_session_store:
        logging.info(f"Creating new Langchain session history in 'llm_chains_session_store' for: {session_id}")
        llm_chains_session_store[session_id] = ChatMessageHistory()
    else:
        logging.info(f"Retrieving existing Langchain session history from 'llm_chains_session_store' for: {session_id}")
    return llm_chains_session_store[session_id]
    
    # --- Persistent Redis Store (Example for Production) ---
    # To use this, you would need a Redis instance and the `redis` and `langchain-redis` packages.
    # redis_url = os.getenv("REDIS_URL")
    # if not redis_url:
    #     raise ValueError("REDIS_URL environment variable not set for persistent session history.")
    # return RedisChatMessageHistory(session_id, url=redis_url)


def define_rag_prompt_template():
    """Defines the prompt template for the RAG chain."""
    template_string = """
    You are an AI assistant specialized in Indian diet and nutrition created by Suprovo.
    Based on the following conversation history and the user's query, provide a simple, practical, and culturally relevant **{dietary_type}** food suggestion suitable for Indian users aiming for **{goal}**.
    If a specific region like **{region}** is mentioned or inferred, prioritize food suggestions from that region.
    Focus on readily available ingredients and common Indian dietary patterns for the specified region.
    Be helpful, encouraging, and specific where possible.
    Use the chat history to understand the context of the user's current query and maintain continuity.
    Strictly adhere to the **{dietary_type}** and **{goal}** requirements, and the **{region}** preference if specified.

    Chat History:
    {chat_history}

    Context from Knowledge Base:
    {context}

    User Query:
    {query}

    {dietary_type} {goal} Food Suggestion (Tailored for {region} Indian context):
    """
    return PromptTemplate(
        template=template_string,
        input_variables=["query", "chat_history", "dietary_type", "goal", "region", "context"]
    )

def setup_qa_chain(llm_gemini: GoogleGenerativeAI, db: Chroma, rag_prompt: PromptTemplate):
    """Sets up the Retrieval Augmented Generation (RAG) chain."""
    try:
        retriever = db.as_retriever(search_kwargs={"k": 5})

        def retrieve_and_log_context(input_dict):
            """Helper to retrieve documents and log their content."""
            docs = retriever.invoke(input_dict["query"])
            if not docs:
                logging.warning(f"No documents retrieved for query: '{input_dict['query']}'")
            context_str = "\n\n".join(doc.page_content for doc in docs)
            logging.info(f"Retrieved Context (snippet): {context_str[:200]}...")
            return context_str

        qa_chain = (
            {
                "context": retrieve_and_log_context,
                "query": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
                "dietary_type": RunnablePassthrough(),
                "goal": RunnablePassthrough(),
                "region": RunnablePassthrough(),
            }
            | rag_prompt
            | llm_gemini
            | StrOutputParser()
        )
        logging.info("Retrieval QA Chain initialized successfully.")
        return qa_chain
    except Exception as e:
        logging.exception("Full QA Chain setup traceback:")
        raise RuntimeError(f"QA Chain setup error: {e}")

def setup_conversational_qa_chain(qa_chain):
    """Wraps the QA chain with message history capabilities."""
    conversational_qa_chain = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    logging.info("Conversational QA Chain initialized.")
    return conversational_qa_chain

def define_merge_prompt_templates():
    """Defines prompt templates for merging RAG and Groq outputs."""
    merge_prompt_default_template = """
    You are an AI assistant specialized in Indian diet and nutrition.
    Your task is to provide a single, coherent, and practical {dietary_type} food suggestion or diet plan for {goal}, tailored for a {region} Indian context.

    Here's the information available:
    {rag_section}
    {additional_suggestions_section}

    Instructions:
    1. Prioritize the "Primary RAG Answer" if it is specific, relevant, and not an error message.
    2. If the "Primary RAG Answer" is generic, insufficient, or indicates an internal system error, then heavily rely on and synthesize from the "Additional Suggestions".
    3. If all sources provide error messages or are unhelpful, state that you couldn't find a specific suggestion and ask the user to rephrase.
    4. Combine information logically and seamlessly, without mentioning the source of each piece.
    5. Ensure the final plan is clear, actionable, and culturally relevant.
    6. If the user's input was only a greeting, respond politely without providing a diet plan.

    Final {dietary_type} {goal} Food Suggestion/Diet Plan (Tailored for {region} Indian context):
    """
    
    merge_prompt_table_template = """
    You are an AI assistant specialized in Indian diet and nutrition.
    Your task is to provide a single, coherent, and practical {dietary_type} food suggestion or diet plan for {goal}, tailored for a {region} Indian context.
    **You MUST present the final diet plan as a clear markdown table. Include columns for Meal, Food Items, and Notes/Considerations.**

    Here's the information available:
    {rag_section}
    {additional_suggestions_section}

    Instructions:
    1. Prioritize the "Primary RAG Answer" if it is specific, relevant, and not an error message.
    2. If the "Primary RAG Answer" is generic, insufficient, or indicates an internal system error, then heavily rely on and synthesize from the "Additional Suggestions".
    3. If all sources provide error messages or are unhelpful, state that you couldn't find a specific suggestion and ask the user to rephrase.
    4. Combine information logically and seamlessly, without mentioning the source of each piece.
    5. Ensure the final plan is clear, actionable, and culturally relevant.
    6. If the user's input was only a greeting, respond politely without providing a diet plan.

    Final {dietary_type} {goal} Diet Plan (Tailored for {region} Indian context, in markdown table format):
    """

    logging.info("Merge Prompt templates created.")
    return (
        PromptTemplate(template=merge_prompt_default_template, input_variables=["rag_section", "additional_suggestions_section", "dietary_type", "goal", "region"]),
        PromptTemplate(template=merge_prompt_table_template, input_variables=["rag_section", "additional_suggestions_section", "dietary_type", "goal", "region"])
    )

# --- Consolidated: Query Analysis and Intent Detection (now primarily for sub-tool use) ---
# These functions are simple heuristics used by the agent to generate tool inputs.
@lru_cache(maxsize=128)
def extract_diet_preference(query: str) -> str:
    q = query.lower()
    if any(x in q for x in ["non-veg", "non veg", "nonvegetarian"]): return "non-vegetarian"
    if "vegan" in q: return "vegan"
    if "veg" in q or "vegetarian" in q: return "vegetarian"
    return "any"

@lru_cache(maxsize=128)
def extract_diet_goal(query: str) -> str:
    q = query.lower()
    if any(p in q for p in ["lose weight", "loss weight", "cut weight", "reduce weight", "lose fat", "cut fat"]): return "weight loss"
    if "gain weight" in q or "weight gain" in q or "muscle gain" in q: return "weight gain"
    if "loss" in q: return "weight loss"
    if "gain" in q: return "weight gain"
    return "diet"

@lru_cache(maxsize=128)
def extract_regional_preference(query: str) -> str:
    q = query.lower()
    if "kolkata" in q or "bengali" in q: return "Bengali" 
    if any(term in q for term in ["south indian", "tamil", "kannada", "telugu", "malayalam", "kanyakumari"]): return "South Indian"
    if any(term in q for term in ["north indian", "punjabi"]): return "North Indian"
    if any(term in q for term in ["west indian", "maharashtrian", "gujarati"]): return "West Indian"
    if any(term in q for term in ["east indian", "odisha", "oriya", "bhubaneswar", "cuttack", "angul"]): return "East Indian"
    return "Indian"

@lru_cache(maxsize=128)
def contains_table_request(query: str) -> bool:
    return any(k in query.lower() for k in ["table", "tabular", "chart", "in a table", "in table format", "as a table"])

def detect_sentiment(llm_instance: GoogleGenerativeAI, query: str) -> str:
    prompt = f"Analyze the sentiment of the following user query. Respond with only one word: 'positive', 'neutral', or 'negative'.\n\nQuery: \"{query}\"\n\nSentiment:"
    try:
        response_obj = llm_instance.invoke(prompt)
        sentiment = (response_obj.content if isinstance(response_obj, AIMessage) else str(response_obj)).strip().lower()
        return sentiment if sentiment in ["positive", "neutral", "negative"] else "neutral"
    except Exception as e:
        logging.error(f"Error detecting sentiment for query '{query}': {e}", exc_info=True)
        return "neutral"

# --- MODIFICATION: Pydantic Models for Tool Input Validation ---
class GenerateDietPlanInput(BaseModel):
    dietary_type: str = Field(default="any", description="e.g., 'vegetarian', 'non-vegetarian', 'vegan', 'any'")
    goal: str = Field(default="diet", description="e.g., 'weight loss', 'weight gain', 'diet'")
    region: str = Field(default="Indian", description="e.g., 'South Indian', 'Punjabi', 'Indian'")
    wants_table: bool = Field(default=False, description="True if user wants a markdown table.")

class ReformatDietPlanInput(BaseModel):
    # MODIFICATION: Agent now passes the content to reformat directly.
    content_to_reformat: str = Field(..., description="The previous AI-generated diet plan text that needs reformatting.")
    wants_table: bool = Field(..., description="The desired format (True for table, False for default text).")
    # Adding these to maintain context for the merge prompt
    dietary_type: str = Field(default="any")
    goal: str = Field(default="diet")
    region: str = Field(default="Indian")

class FetchRecipeInput(BaseModel):
    recipe_name: str = Field(..., min_length=3, description="The name of the recipe to fetch, e.g., 'Dal Makhani'.")

class LookupNutritionInput(BaseModel):
    food_item: str = Field(..., min_length=3, description="The food item to look up, e.g., 'rice'.")


# --- New: Placeholder Tools for Agent ---
async def tool_fetch_recipe(recipe_name: str) -> str:
    """Placeholder tool to simulate fetching a recipe."""
    logging.info(f"Executing tool: fetch_recipe for '{recipe_name}'")
    await asyncio.sleep(0.5) 
    # ... (rest of the tool logic is unchanged)
    if "dal makhani" in recipe_name.lower():
        return f"Recipe for {recipe_name}: Ingredients - Black lentils, kidney beans, butter, cream, tomatoes, ginger-garlic paste. Steps - Soak, boil, temper, simmer. Serve hot with naan or rice."
    elif "paneer tikka" in recipe_name.lower():
        return f"Recipe for {recipe_name}: Ingredients - Paneer, yogurt, ginger-garlic paste, spices, bell peppers, onions. Steps - Marinate, skewer, grill/bake."
    elif "chicken tikka masala" in recipe_name.lower(): 
        return f"Recipe for {recipe_name}: Ingredients - Chicken, yogurt, ginger-garlic paste, spices, tomatoes, cream. Steps - Marinate chicken, grill/bake, cook in a rich tomato-cream sauce. Serve with naan/rice."
    else:
        return f"Recipe for {recipe_name}: Detailed recipe unavailable, but typically involves [basic ingredients] and [basic cooking method]."

async def tool_lookup_nutrition_facts(food_item: str) -> str:
    """Placeholder tool to simulate looking up nutrition facts."""
    logging.info(f"Executing tool: lookup_nutrition_facts for '{food_item}'")
    await asyncio.sleep(0.5)
    # ... (rest of the tool logic is unchanged)
    clean_food_item = food_item.lower().strip()
    if "rice" in clean_food_item:
        return f"Nutrition facts for {food_item} (per 100g cooked): Calories: 130, Carbs: 28g, Protein: 2.7g, Fat: 0.3g."
    elif "lentils" in clean_food_item:
        return f"Nutrition facts for {food_item} (per 100g cooked): Calories: 116, Carbs: 20g, Protein: 9g, Fat: 0.4g. Rich in fiber."
    elif "avocado" in clean_food_item:
        return f"Nutrition facts for {food_item} (per 100g): Calories: 160, Fat: 14.7g (mostly healthy monounsaturated), Carbs: 8.5g, Protein: 2g."
    elif "pineapple" in clean_food_item: 
        return f"Nutrition facts for {food_item} (per 100g): Calories: 50, Carbs: 13g, Protein: 0.5g, Fat: 0.1g. Rich in Vitamin C and Manganese."
    elif "non veg vs veg" in clean_food_item or "non-veg vs veg" in clean_food_item or "non-vegetarian vs vegetarian" in clean_food_item:
        return ("Comparing non-vegetarian vs. vegetarian nutrition:\n\n"
            "**Non-Vegetarian (e.g., Chicken Breast - 100g cooked):**\n"
            "  - Calories: ~165 kcal\n"
            "  - Protein: ~31g (complete protein)\n"
            "  - Fat: ~3.6g (low in saturated fat if skinless)\n"
            "  - Key nutrients: B vitamins, iron, zinc.\n\n"
            "**Vegetarian (e.g., Cooked Lentils - 100g):**\n"
            "  - Calories: ~116 kcal\n"
            "  - Protein: ~9g (incomplete, but can be complete with grains)\n"
            "  - Fat: ~0.4g (very low)\n"
            "  - Key nutrients: Fiber, folate, potassium, iron (non-heme).\n\n"
            "**Vegetarian (e.g., Paneer - 100g):**\n"
            "  - Calories: ~265 kcal\n"
            "  - Protein: ~18g\n"
            "  - Fat: ~20g (higher in saturated fat)\n"
            "  - Key nutrients: Calcium, Vitamin D.\n\n"
            "Overall, non-vegetarian options often provide complete proteins and higher iron/B12, "
            "while vegetarian diets excel in fiber, diverse micronutrients (from varied plant sources), "
            "and can be lower in saturated fat if plant-based proteins are prioritized. Careful planning ensures adequate nutrition in both.")
    elif "milk" in clean_food_item or "benefits of drinking milk" in clean_food_item:
        return ("Benefits of drinking milk:\n"
            "- **Strong Bones and Teeth:** Rich in calcium, phosphorus, and Vitamin D (often fortified), essential for bone health and preventing osteoporosis.\n"
            "- **Muscle Growth and Repair:** Contains high-quality protein (whey and casein), which aids in muscle building and post-exercise recovery.\n"
            "- **Weight Management:** Protein content can help with satiety, potentially reducing overall calorie intake. Some studies suggest a link to healthy weight.\n"
            "- **Heart Health:** Fortified milk with Vitamin D and calcium may contribute to blood pressure regulation. However, choose low-fat options for better heart health.\n"
            "- **Hydration:** Milk is primarily water, contributing to overall hydration.\n"
            "- **Nutrient-Dense:** Provides a good source of B vitamins (B12, riboflavin), potassium, and other essential minerals.")
    else:
        return f"Nutrition facts for {food_item}: Calories, carbs, protein, and fat vary. Generally healthy. For specific data, please specify a single, common food item."

# --- Agentic Orchestration Pydantic Model ---
class AgentAction(BaseModel):
    """Represents the action the AI Agent decides to take or the final answer it provides."""
    thought: str = Field(..., description="A brief thought process explaining the current decision.")
    tool_name: Optional[str] = Field(None, description="The name of the tool to use. Must be one of: 'generate_diet_plan', 'reformat_diet_plan', 'handle_greeting', 'handle_identity', 'fetch_recipe', 'lookup_nutrition_facts'.")
    tool_input: Optional[Dict[str, Any]] = Field(None, description="A dictionary of parameters for the selected tool.")
    final_answer: Optional[str] = Field(None, description="The final answer to the user's request. Only set this if the task is complete.")

# --- MODIFICATION: Updated Agentic Orchestrator Prompt ---
ORCHESTRATOR_PROMPT_TEMPLATE = """
You are an intelligent AI agent named AAHAR, specialized in Indian diet and nutrition.
Your goal is to assist users with their diet-related queries by thinking step-by-step, deciding on appropriate actions, and providing comprehensive answers.
You must always respond with a JSON object that adheres to the `AgentAction` Pydantic model.
Your response MUST include a `thought` describing your reasoning.
You must either provide a `final_answer` OR specify a `tool_name` and `tool_input`.
DO NOT use both `tool_name` and `final_answer` in the same response.

Available Tools:
1.  **`handle_greeting`**:
    * **Description**: Respond to simple greetings (e.g., "Hi", "Hello", "Namaste").
    * **Input**: None needed.
    * **When to use**: If the user's query is purely a greeting.
2.  **`handle_identity`**:
    * **Description**: Respond to queries about your identity (e.g., "Who are you?").
    * **Input**: None needed.
    * **When to use**: If the user's query is about your identity.
3.  **`reformat_diet_plan`**:
    * **Description**: Reformat a *previous* diet plan provided by the AI.
    * **Input**: `content_to_reformat: string` (The full text of the previous AI message to reformat), `wants_table: boolean`.
    * **When to use**: Only if there is a previous AI message in the `chat_history` that looks like a diet plan, AND the user is explicitly asking to reformat it (e.g., "put that in a table"). You must extract the relevant AI message from the history and provide it as `content_to_reformat`.
4.  **`generate_diet_plan`**:
    * **Description**: Generate a new diet suggestion or detailed diet plan.
    * **Input**: `dietary_type: string`, `goal: string`, `region: string`, `wants_table: boolean`.
    * **When to use**: For most diet-related queries that require generating a new plan or suggestion. Infer parameters from the user query.
5.  **`fetch_recipe`**:
    * **Description**: Fetch a simple recipe for a given food item.
    * **Input**: `recipe_name: string`.
    * **When to use**: If the user explicitly asks for a "recipe for X" or "how to make Y".
6.  **`lookup_nutrition_facts`**:
    * **Description**: Look up basic nutrition facts for a given food item.
    * **Input**: `food_item: string`.
    * **When to use**: If the user asks for "nutrition facts", "calories in", "protein in", or a comparison like "non veg vs veg nutrition".

**Agent's State:**
- Current User Query: "{query}"
- User's Sentiment: "{sentiment}" (You can use this to adjust your tone)
- Chat History:
{chat_history}
- Agent Scratchpad (Observations from previous tool executions):
{agent_scratchpad}

Think step-by-step. What is the user's ultimate goal? What is the next logical step?
If a tool has been executed and its output in the scratchpad directly answers the query, set `final_answer` to that output and stop.
Otherwise, select the next `tool_name` and `tool_input`.

Output (JSON adhering to AgentAction Pydantic model):
"""

ORCHESTRATOR_PROMPT = PromptTemplate(
    template=ORCHESTRATOR_PROMPT_TEMPLATE,
    input_variables=["chat_history", "query", "agent_scratchpad", "sentiment"],
)

# --- Main FastAPI Application Setup ---

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FASTAPI_SECRET_KEY = os.getenv("FASTAPI_SECRET_KEY", "a_very_secure_random_key_CHANGE_THIS_IN_PRODUCTION")
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "Diet Suggest Logs")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('langchain_community').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('langchain_chroma').setLevel(logging.WARNING)


# --- Google Sheets Setup ---
sheet = None
sheet_enabled = False
try:
    if GOOGLE_CREDS_JSON:
        creds_dict = json.loads(base64.b64decode(GOOGLE_CREDS_JSON.strip()).decode('utf-8')) 
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            creds_dict,
            ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        )
        gs_client = gspread.authorize(creds)
        sheet = gs_client.open(GOOGLE_SHEET_NAME).sheet1
        sheet_enabled = True
        logging.info("✅ Google Sheets connected for logging.")
    else:
        logging.warning("⚠️ GOOGLE_CREDS_JSON environment variable not set. Google Sheets disabled.")
except Exception as e:
    logging.warning(f"⚠️ Google Sheets connection failed: {e}. Logging to sheet disabled.", exc_info=True)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Indian Diet Recommendation API",
    description="A backend API for personalized Indian diet suggestions using RAG and LLMs.",
    version="0.4.0", # MODIFICATION: Version bump
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # MODIFICATION: Should be restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=FASTAPI_SECRET_KEY)

# --- Global Variables for Initialized Components ---
llm_gemini: Optional[GoogleGenerativeAI] = None
llm_orchestrator: Optional[GoogleGenerativeAI] = None
db: Optional[Chroma] = None
conversational_qa_chain: Optional[Any] = None
merge_prompt_default: Optional[PromptTemplate] = None
merge_prompt_table: Optional[PromptTemplate] = None
orchestrator_chain: Optional[Any] = None

@app.on_event("startup")
async def startup_event():
    """Initializes LLMs, DB, and LangChain components at startup."""
    global llm_gemini, llm_orchestrator, db, conversational_qa_chain, \
           merge_prompt_default, merge_prompt_table, orchestrator_chain

    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    
    llm_gemini = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.5)
    llm_orchestrator = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.1)
    logging.info("✅ Gemini LLMs initialized.")

    download_and_extract_db_for_app()
    db, _ = setup_vector_database(chroma_db_directory="/tmp/chroma_db")
    
    rag_prompt = define_rag_prompt_template()
    qa_chain = setup_qa_chain(llm_gemini, db, rag_prompt)
    conversational_qa_chain = setup_conversational_qa_chain(qa_chain)
    merge_prompt_default, merge_prompt_table = define_merge_prompt_templates()

    def parse_agent_action_output(llm_output: Union[AIMessage, str]) -> AgentAction:
        """Parses the LLM JSON output into an AgentAction Pydantic model, handling common issues."""
        parser = JsonOutputParser(pydantic_object=AgentAction)
        content_str = llm_output.content if isinstance(llm_output, AIMessage) else str(llm_output)
        
        json_match = re.search(r"```json\n(.*)\n```", content_str, re.DOTALL)
        json_str = json_match.group(1).strip() if json_match else content_str.strip()

        try:
            # First attempt to parse directly
            parsed_data = json.loads(json_str)
            # Handle cases where the agent might nest its action
            if isinstance(parsed_data, dict) and "agent_action" in parsed_data:
                return AgentAction(**parsed_data["agent_action"])
            return AgentAction(**parsed_data)
        except (ValidationError, json.JSONDecodeError) as e:
            logging.error(f"Validation/JSON error parsing AgentAction: {e}. Raw output: {json_str}")
            # Fallback for malformed output
            return AgentAction(
                thought="Error parsing my own output. I need to regenerate my decision.",
                final_answer="I seem to have encountered an internal formatting error. Could you please try rephrasing your request?"
            )

    orchestrator_chain = ORCHESTRATOR_PROMPT | llm_orchestrator | RunnableLambda(parse_agent_action_output)
    logging.info("✅ LangChain components initialized successfully.")

# --- Pydantic Schema for Request Body ---
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

# --- API Endpoints ---

@app.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    """Main chat endpoint with an improved LLM-driven agent loop."""
    user_query = chat_request.query
    session_id = chat_request.session_id or request.session.get("session_id") or f"session_{os.urandom(8).hex()}"
    request.session["session_id"] = session_id

    logging.info(f"📩 Query: '{user_query}' | Session: {session_id}")

    if not all([llm_gemini, llm_orchestrator, conversational_qa_chain]):
        raise HTTPException(status_code=503, detail="AI services are not ready. Please try again later.")

    # MODIFICATION: Sentiment detection moved before the agent loop
    sentiment = detect_sentiment(llm_orchestrator, user_query)
    logging.info(f"Sentiment for query '{user_query}': {sentiment}")

    chat_history_lc = get_session_history(session_id).messages
    formatted_chat_history = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in chat_history_lc])

    # --- AGENT LOOP ---
    max_agent_iterations = 5
    agent_scratchpad: List[Dict[str, Any]] = []
    response_text = "I'm sorry, I couldn't process your request. Please try again."

    try:
        # MODIFICATION: Added a 30-second timeout for the entire agent loop
        async with asyncio.timeout(30):
            for i in range(max_agent_iterations):
                logging.info(f"🔄 Agent Iteration {i+1}/{max_agent_iterations}")
                
                scratchpad_str = "\n".join([f"Tool: {item.get('tool_name')}, Input: {item.get('tool_input')}, Output: {item.get('tool_output')}" for item in agent_scratchpad])

                orchestrator_decision = await orchestrator_chain.ainvoke({
                    "query": user_query,
                    "chat_history": formatted_chat_history,
                    "agent_scratchpad": scratchpad_str,
                    "sentiment": sentiment
                }, config={"callbacks": [SafeTracer()], "configurable": {"session_id": session_id}})
                
                logging.info(f"✨ Orchestrator (Iter {i+1}): Thought='{orchestrator_decision.thought}' Tool='{orchestrator_decision.tool_name}'")

                if orchestrator_decision.final_answer:
                    response_text = orchestrator_decision.final_answer
                    logging.info(f"✅ Agent provided final answer on iteration {i+1}.")
                    break
                
                if not orchestrator_decision.tool_name:
                    response_text = "I'm not sure how to proceed. Could you please rephrase your request?"
                    logging.warning("Agent decided on no tool and no final answer.")
                    break

                tool_name = orchestrator_decision.tool_name
                tool_input_dict = orchestrator_decision.tool_input or {}
                tool_output = f"Error: Tool '{tool_name}' execution failed unexpectedly."

                try:
                    # --- TOOL EXECUTION ---
                    if tool_name == "handle_greeting":
                        response_text = "Namaste! How can I assist you with a healthy Indian diet today?"
                        break
                    elif tool_name == "handle_identity":
                        response_text = "I am AAHAR, an AI assistant specialized in Indian diet and nutrition, created by Suprovo."
                        break
                    
                    elif tool_name == "generate_diet_plan":
                        validated_input = GenerateDietPlanInput(**tool_input_dict)
                        rag_output = await conversational_qa_chain.ainvoke({
                            "query": user_query, "chat_history": chat_history_lc, **validated_input.model_dump()
                        }, config={"callbacks": [SafeTracer()], "configurable": {"session_id": session_id}})
                        
                        groq_output = await cached_groq_answers_async(user_query, GROQ_API_KEY, **validated_input.model_dump())
                        
                        merge_prompt = merge_prompt_table if validated_input.wants_table else merge_prompt_default
                        merge_input = {
                            "rag_section": f"Primary RAG Answer:\n{rag_output}",
                            "additional_suggestions_section": "\n".join([f"- {k.title()} Suggestion: {v}" for k, v in groq_output.items()]),
                            **validated_input.model_dump()
                        }
                        merged_result = await llm_gemini.ainvoke(merge_prompt.format(**merge_input), config={"callbacks": [SafeTracer()]})
                        tool_output = merged_result.content if isinstance(merged_result, AIMessage) else str(merged_result)
                    
                    elif tool_name == "reformat_diet_plan":
                        validated_input = ReformatDietPlanInput(**tool_input_dict)
                        merge_prompt = merge_prompt_table if validated_input.wants_table else merge_prompt_default
                        reformat_result = await llm_gemini.ainvoke(
                            merge_prompt.format(
                                rag_section=f"Previous Answer to Reformat:\n{validated_input.content_to_reformat}",
                                additional_suggestions_section="No new suggestions needed, just reformat the above.",
                                **validated_input.model_dump()
                            ), config={"callbacks": [SafeTracer()]})
                        tool_output = reformat_result.content if isinstance(reformat_result, AIMessage) else str(reformat_result)

                    elif tool_name == "fetch_recipe":
                        validated_input = FetchRecipeInput(**tool_input_dict)
                        tool_output = await tool_fetch_recipe(validated_input.recipe_name)
                    
                    elif tool_name == "lookup_nutrition_facts":
                        validated_input = LookupNutritionInput(**tool_input_dict)
                        tool_output = await tool_lookup_nutrition_facts(validated_input.food_item)
                    
                    else:
                        tool_output = f"Error: Unknown tool '{tool_name}' requested."

                except ValidationError as e:
                    tool_output = f"Error: My attempt to use the '{tool_name}' tool had invalid parameters: {e}. I will try to correct this."
                    logging.warning(tool_output)
                except Exception as e:
                    tool_output = f"Error during execution of tool '{tool_name}': {e}"
                    logging.error(tool_output, exc_info=True)
                
                agent_scratchpad.append({"tool_name": tool_name, "tool_input": tool_input_dict, "tool_output": tool_output})

        else: # This 'else' belongs to the 'for' loop
            logging.warning(f"Agent loop finished after {max_agent_iterations} iterations without a final answer.")
            response_text = "I'm having trouble finalizing my response. Please try rephrasing your request."

    except asyncio.TimeoutError:
        logging.error(f"Agent loop timed out for session {session_id}.")
        response_text = "I'm sorry, your request took too long to process. Please try again."
    except Exception as e:
        logging.error(f"❌ Global error in /chat endpoint for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

    get_session_history(session_id).add_user_message(user_query)
    get_session_history(session_id).add_ai_message(response_text)

    if sheet_enabled and sheet:
        try:
            sheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session_id, user_query, response_text, sentiment])
        except Exception as e:
            logging.warning(f"⚠️ Google Sheet logging failed: {e}")

    return JSONResponse(content={"answer": response_text, "session_id": session_id})

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "✅ Indian Diet Recommendation API is running. Use POST /chat to interact."}

if __name__ == "__main__":
    import uvicorn
    logging.info("🚀 Starting FastAPI application locally...")
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)), reload=True)
