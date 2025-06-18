# qa_chain.py
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
# Removed BaseChatMessageHistory and related imports, as we're using InMemoryChatMessageHistory directly now
from langchain_community.chat_message_histories import ChatMessageHistory # For in-memory history
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage # Still useful for type hints in history
import logging

# Centralized in-memory store for chat histories
# This dict holds ChatMessageHistory objects, keyed by session_id
STORE = {}

def get_session_history_in_memory(session_id: str) -> ChatMessageHistory:
    """
    Factory function to get or create an InMemoryChatMessageHistory instance.
    This function will be used by RunnableWithMessageHistory.
    """
    if session_id not in STORE:
        STORE[session_id] = ChatMessageHistory()
        logging.info(f"Initialized new in-memory chat history for session ID: {session_id}")
    return STORE[session_id]

# MODIFIED: create_qa_chain now accepts the specific prompt template
def create_qa_chain(llm: BaseChatModel, retriever: VectorStoreRetriever, rag_prompt: PromptTemplate):
    """
    Creates the core QA chain for RAG.
    """
    logging.info("Creating QA chain using custom prompt and StuffDocumentsChain.")
    doc_chain = LLMChain(llm=llm, prompt=rag_prompt) # Use the provided rag_prompt
    chain = StuffDocumentsChain(
        llm_chain=doc_chain,
        document_variable_name="context"
    )
    # ConversationalRetrievalChain handles the question rephrasing and combines with docs
    return ConversationalRetrievalChain(
        question_generator=LLMChain(llm=llm, prompt=PromptTemplate.from_template("Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Question: {question}\nStandalone question:")),
        retriever=retriever,
        combine_docs_chain=chain,
        return_source_documents=True
    )

# MODIFIED: setup_conversational_qa_chain no longer takes credentials_json
def setup_conversational_qa_chain(base_qa_chain: ConversationalRetrievalChain):
    """
    Sets up the conversational QA chain using RunnableWithMessageHistory
    with the in-memory chat history factory.
    """
    logging.info("Setting up conversational chain with in-memory message history.")
    return RunnableWithMessageHistory(
        base_qa_chain,
        get_session_history_in_memory, # Use the in-memory history factory
        input_messages_key="question", # Key for the user's question
        history_messages_key="chat_history" # Key for chat history in the prompt
    )

