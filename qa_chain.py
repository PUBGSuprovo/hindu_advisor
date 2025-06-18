# qa_chain.py
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory # Correct base class
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage # For message types
from prompts import scripture_prompt, merge_prompt_default, merge_prompt_table
import logging
import gspread # For Google Sheets interaction
from google.oauth2 import service_account
import json
import os

# Define the sheet and credentials globally for the class
SPREADSHEET_ID = "1MS-6RNx8N0uKnOzunyVNjBFiSUysibCdS8HBk_uyYik" # Your provided spreadsheet ID

class GoogleSheetChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Google Sheet."""

    def __init__(self, session_id: str, credentials_json: str):
        self.session_id = session_id
        self.client = self._authenticate_gspread(credentials_json)
        self.sheet = self._get_or_create_sheet()
        logging.info(f"GoogleSheetChatMessageHistory initialized for session: {session_id}")

    def _authenticate_gspread(self, credentials_json: str):
        """Authenticates with Google Sheets API using service account credentials."""
        try:
            creds_info = json.loads(credentials_json)
            creds = service_account.Credentials.from_service_account_info(creds_info)
            client = gspread.authorize(creds)
            return client
        except Exception as e:
            logging.critical(f"Error authenticating with Google Sheets: {e}")
            raise

    def _get_or_create_sheet(self):
        """Gets the specified worksheet or creates it if it doesn't exist."""
        try:
            spreadsheet = self.client.open_by_key(SPREADSHEET_ID)
            try:
                worksheet = spreadsheet.worksheet(self.session_id)
                logging.info(f"Found existing worksheet for session: {self.session_id}")
                return worksheet
            except gspread.exceptions.WorksheetNotFound:
                # Create sheet if not found
                worksheet = spreadsheet.add_worksheet(title=self.session_id, rows=1000, cols=5)
                # Add headers
                worksheet.append_row(["Timestamp", "Session ID", "Sender", "Message Content"])
                logging.info(f"Created new worksheet for session: {self.session_id}")
                return worksheet
        except gspread.exceptions.SpreadsheetNotFound:
            logging.critical(f"Spreadsheet with ID '{SPREADSHEET_ID}' not found. "
                             "Please ensure the ID is correct and the service account has access.")
            raise
        except Exception as e:
            logging.critical(f"Error accessing or creating worksheet: {e}")
            raise


    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieve all messages from the Google Sheet for the current session."""
        rows = self.sheet.get_all_values()
        if not rows or len(rows) <= 1: # Skip header row
            return []
        
        messages = []
        for row in rows[1:]: # Start from the second row to skip headers
            if len(row) >= 4: # Ensure row has enough columns
                timestamp, session_id_col, sender, content = row[0], row[1], row[2], row[3]
                if sender.lower() == "human":
                    messages.append(HumanMessage(content=content))
                elif sender.lower() == "ai":
                    messages.append(AIMessage(content=content))
            else:
                logging.warning(f"Skipping malformed row in Google Sheet for session {self.session_id}: {row}")
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the Google Sheet."""
        timestamp = gspread.utils.get_iso_string()
        sender = "Human" if isinstance(message, HumanMessage) else "AI"
        content = message.content
        try:
            self.sheet.append_row([timestamp, self.session_id, sender, content])
            logging.info(f"Message added to Google Sheet for session {self.session_id}: {sender} - {content[:50]}...")
        except Exception as e:
            logging.error(f"Error adding message to Google Sheet: {e}")

    def clear(self) -> None:
        """Clear all messages for the current session by deleting the worksheet."""
        try:
            spreadsheet = self.client.open_by_key(SPREADSHEET_ID)
            spreadsheet.del_worksheet(self.sheet)
            logging.info(f"Cleared session history by deleting worksheet: {self.session_id}")
            # Re-initialize the sheet to ensure it's ready for new messages
            self.sheet = self._get_or_create_sheet()
        except gspread.exceptions.WorksheetNotFound:
            logging.warning(f"Attempted to clear non-existent worksheet for session {self.session_id}.")
        except Exception as e:
            logging.error(f"Error clearing session in Google Sheet: {e}")

# This will now be a factory function that creates a GoogleSheetChatMessageHistory instance
def get_session_history(session_id: str, credentials_json: str) -> GoogleSheetChatMessageHistory:
    """
    Factory function to get or create a GoogleSheetChatMessageHistory instance.
    The credentials_json will be passed from FastAPI app.
    """
    # In a real-world scenario, you might want to cache these instances if performance is an issue
    # But for simplicity and to ensure fresh auth, we create one per request.
    logging.info(f"Requesting session history for session ID: {session_id}")
    return GoogleSheetChatMessageHistory(session_id, credentials_json)

def create_qa_chain(llm: BaseChatModel, retriever: VectorStoreRetriever):
    """
    Creates the core QA chain for RAG.
    """
    logging.info("Creating QA chain using custom prompt and StuffDocumentsChain.")
    doc_chain = LLMChain(llm=llm, prompt=scripture_prompt)
    chain = StuffDocumentsChain(
        llm_chain=doc_chain,
        document_variable_name="context"
    )
    return ConversationalRetrievalChain(
        question_generator=LLMChain(llm=llm, prompt=PromptTemplate.from_template("Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Question: {question}\nStandalone question:")),
        retriever=retriever,
        combine_docs_chain=chain,
        return_source_documents=True
    )

def create_conversational_chain(llm: BaseChatModel, retriever: VectorStoreRetriever, credentials_json: str):
    """
    Creates a conversational chain with message history using Google Sheets.
    """
    logging.info("Creating conversational chain with message history via Google Sheets.")
    
    base_qa_chain = create_qa_chain(llm, retriever)

    # Lambda function to pass credentials to get_session_history
    return RunnableWithMessageHistory(
        base_qa_chain,
        lambda session_id: get_session_history(session_id, credentials_json), # Pass credentials
        input_messages_key="question",
        history_messages_key="chat_history"
    )

async def merge_groq_and_rag_answers(
    llm: BaseChatModel,
    rag_answer: str,
    groq_suggestions: dict,
    spiritual_concept: str,
    life_problem: str,
    scripture_source: str,
    format_as_table: bool = False
) -> str:
    """
    Merges the RAG answer with suggestions from Groq models using an LLM.

    Args:
        llm (BaseChatModel): The LLM to use for merging.
        rag_answer (str): The answer from the RAG chain.
        groq_suggestions (dict): A dictionary of suggestions from Groq models.
        spiritual_concept (str): The detected spiritual concept.
        life_problem (str): The detected life problem.
        scripture_source (str): The detected scripture source.
        format_as_table (bool): Whether to format the final output as a markdown table.

    Returns:
        str: The merged and refined answer.
    """
    logging.info(f"Merging RAG answer and Groq suggestions. Table format requested: {format_as_table}")
    
    if format_as_table:
        merge_prompt = merge_prompt_table
        logging.info("Using table format merge prompt.")
    else:
        merge_prompt = merge_prompt_default
        logging.info("Using default merge prompt.")

    merge_chain = LLMChain(llm=llm, prompt=merge_prompt)

    try:
        merged_result = await merge_chain.ainvoke({
            "rag": rag_answer,
            "llama": groq_suggestions.get("llama", "N/A"),
            "mixtral": groq_suggestions.get("mixtral", "N/A"),
            "gemma": groq_suggestions.get("gemma", "N/A"),
            "spiritual_concept": spiritual_concept,
            "life_problem": life_problem,
            "scripture_source": scripture_source
        })
        return merged_result['text']
    except Exception as e:
        logging.error(f"Error during answer merging: {e}", exc_info=True)
        return f"Error merging answers: {e}. Here is the RAG answer: {rag_answer}"
