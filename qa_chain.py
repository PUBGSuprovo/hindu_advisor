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
from prompts import scripture_prompt, merge_prompt_default, merge_prompt_table # Import merge prompts
import logging
from functools import partial # NEW: Import partial for passing credentials
import gspread # For Google Sheets interaction
from google.oauth2 import service_account
import json
import os

# Define the sheet ID globally for the GoogleSheetChatMessageHistory class
# IMPORTANT: Replace with your actual Google Sheet ID where chat logs will be stored
SPREADSHEET_ID = "1MS-6RNx8N0uKnOzunyVNjBFiSUysibCdS8HBk_uyYik"

class GoogleSheetChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Google Sheet."""

    def __init__(self, session_id: str, credentials_json: str):
        self.session_id = session_id
        # Client will be authorized once per instance creation
        self.client = self._authenticate_gspread(credentials_json)
        # Sheet will be fetched/created once per instance creation
        self.sheet = self._get_or_create_sheet()
        logging.info(f"GoogleSheetChatMessageHistory initialized for session: {session_id}")

    def _authenticate_gspread(self, credentials_json: str):
        """Authenticates with Google Sheets API using service account credentials."""
        if not credentials_json:
            raise ValueError("Google Credentials JSON string is empty.")
        try:
            creds_info = json.loads(credentials_json)
            creds = service_account.Credentials.from_service_account_info(creds_info)
            client = gspread.authorize(creds)
            return client
        except json.JSONDecodeError:
            logging.critical("Error: GOOGLE_CREDS_JSON is not a valid JSON string.", exc_info=True)
            raise ValueError("Invalid GOOGLE_CREDS_JSON format.")
        except Exception as e:
            logging.critical(f"Error authenticating with Google Sheets: {e}", exc_info=True)
            raise

    def _get_or_create_sheet(self):
        """Gets the specified worksheet or creates it if it doesn't exist."""
        try:
            # Open the main spreadsheet by its ID
            spreadsheet = self.client.open_by_key(SPREADSHEET_ID)
            try:
                # Try to get the worksheet for the current session_id
                worksheet = spreadsheet.worksheet(self.session_id)
                logging.info(f"Found existing worksheet for session: {self.session_id}")
                return worksheet
            except gspread.exceptions.WorksheetNotFound:
                # If worksheet not found, create a new one with the session_id as title
                worksheet = spreadsheet.add_worksheet(title=self.session_id, rows=1000, cols=5)
                # Add headers to the new worksheet
                worksheet.append_row(["Timestamp", "Session ID", "Sender", "Message Content"])
                logging.info(f"Created new worksheet for session: {self.session_id}")
                return worksheet
        except gspread.exceptions.SpreadsheetNotFound:
            logging.critical(f"Spreadsheet with ID '{SPREADSHEET_ID}' not found. "
                             "Please ensure the ID is correct and the service account has access.", exc_info=True)
            raise
        except Exception as e:
            logging.critical(f"Error accessing or creating worksheet: {e}", exc_info=True)
            raise

    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieve all messages from the Google Sheet for the current session."""
        try:
            rows = self.sheet.get_all_values()
            if not rows or len(rows) <= 1: # Skip header row or if sheet is empty
                return []
                
            messages = []
            for row in rows[1:]: # Start from the second row to skip headers
                if len(row) >= 4: # Ensure row has enough columns (Timestamp, Session ID, Sender, Message Content)
                    # timestamp, session_id_col, sender, content = row[0], row[1], row[2], row[3]
                    sender = row[2] # Sender is in the 3rd column (index 2)
                    content = row[3] # Message Content is in the 4th column (index 3)
                    
                    if sender.lower() == "human":
                        messages.append(HumanMessage(content=content))
                    elif sender.lower() == "ai":
                        messages.append(AIMessage(content=content))
                    else:
                        logging.warning(f"Unknown sender type '{sender}' in Google Sheet row for session {self.session_id}.")
                else:
                    logging.warning(f"Skipping malformed row in Google Sheet for session {self.session_id}: {row}")
            logging.info(f"Loaded {len(messages)} messages from Google Sheet for session {self.session_id}.")
            return messages
        except Exception as e:
            logging.error(f"Error retrieving messages from Google Sheet for session {self.session_id}: {e}", exc_info=True)
            return [] # Return empty list on error to prevent application crash

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the Google Sheet."""
        timestamp = gspread.utils.get_iso_string() # Get ISO formatted timestamp
        sender = "Human" if isinstance(message, HumanMessage) else "AI"
        content = message.content # Extract message content
        try:
            self.sheet.append_row([timestamp, self.session_id, sender, content])
            logging.info(f"Message added to Google Sheet for session {self.session_id}: {sender} - {content[:50]}...")
        except Exception as e:
            logging.error(f"Error adding message to Google Sheet for session {self.session_id}: {e}", exc_info=True)

    def clear(self) -> None:
        """Clear all messages for the current session by deleting the worksheet."""
        try:
            spreadsheet = self.client.open_by_key(SPREADSHEET_ID)
            spreadsheet.del_worksheet(self.sheet) # Delete the current worksheet
            logging.info(f"Cleared session history by deleting worksheet: {self.session_id}")
            # Re-initialize the sheet to ensure it's ready for new messages immediately
            self.sheet = self._get_or_create_sheet()
        except gspread.exceptions.WorksheetNotFound:
            logging.warning(f"Attempted to clear non-existent worksheet for session {self.session_id}.")
        except Exception as e:
            logging.error(f"Error clearing session in Google Sheet: {e}", exc_info=True)

# Factory function to get or create a GoogleSheetChatMessageHistory instance
def get_session_history(session_id: str, credentials_json: str) -> GoogleSheetChatMessageHistory:
    """
    Factory function to get or create a GoogleSheetChatMessageHistory instance.
    This function is passed to RunnableWithMessageHistory.
    """
    logging.info(f"Requesting session history manager for session ID: {session_id}")
    return GoogleSheetChatMessageHistory(session_id, credentials_json)

# MODIFIED: create_qa_chain now returns a ConversationalRetrievalChain directly
def create_qa_chain(llm: BaseChatModel, retriever: VectorStoreRetriever):
    """
    Creates the core QA chain for RAG.
    This chain uses a custom prompt template for Hindu scriptures.
    """
    logging.info("Creating QA chain using custom prompt and StuffDocumentsChain.")
    doc_chain = LLMChain(llm=llm, prompt=scripture_prompt) # Uses the scripture_prompt from prompts.py
    chain = StuffDocumentsChain(
        llm_chain=doc_chain,
        document_variable_name="context" # The variable name expected by the prompt for retrieved docs
    )
    # ConversationalRetrievalChain is designed for chat-like interactions with RAG
    return ConversationalRetrievalChain(
        question_generator=LLMChain(llm=llm, prompt=PromptTemplate.from_template("Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Question: {question}\nStandalone question:")),
        retriever=retriever,
        combine_docs_chain=chain,
        return_source_documents=True # Optionally return source documents if needed for UI
    )

# MODIFIED: create_conversational_chain now accepts google_credentials_json
def create_conversational_chain(llm: BaseChatModel, retriever: VectorStoreRetriever, google_credentials_json: str):
    """
    Creates a conversational chain with message history using Google Sheets.
    This wraps the base QA chain with RAG and chat memory.
    """
    logging.info("Creating conversational chain with message history via Google Sheets.")
    
    base_qa_chain = create_qa_chain(llm, retriever)

    # Use functools.partial to create a new function that pre-fills the credentials_json argument
    # This partial function is then passed to RunnableWithMessageHistory, which will call it with just session_id
    get_history_for_run = partial(get_session_history, google_credentials_json=google_credentials_json)

    # RunnableWithMessageHistory manages conversation state and passes it to the underlying chain
    return RunnableWithMessageHistory(
        base_qa_chain,
        get_history_for_run, # This function will be called to get the chat history for a session
        input_messages_key="question", # Key for the user's current question in the input dictionary
        history_messages_key="chat_history" # Key for where the chat history should be injected into the chain's input
    )
