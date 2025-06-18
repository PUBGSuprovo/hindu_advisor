# qa_chain.py
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from prompts import scripture_prompt # Assuming scripture_prompt is defined here or imported
import logging

store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        logging.info(f"Initialized new chat history for session ID: {session_id}")
    return store[session_id]

# MODIFIED: create_qa_chain now returns a ConversationalRetrievalChain directly
def create_qa_chain(llm: BaseChatModel, retriever: VectorStoreRetriever):
    logging.info("Creating QA chain using custom prompt and StuffDocumentsChain.")
    doc_chain = LLMChain(llm=llm, prompt=scripture_prompt)
    chain = StuffDocumentsChain(
        llm_chain=doc_chain,
        document_variable_name="context"
    )
    # Return a ConversationalRetrievalChain directly.
    # This chain will then be wrapped by RunnableWithMessageHistory in create_conversational_chain.
    return ConversationalRetrievalChain(
        question_generator=LLMChain(llm=llm, prompt=PromptTemplate.from_template("Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Question: {question}\nStandalone question:")),
        retriever=retriever,
        combine_docs_chain=chain,
        return_source_documents=True
    )

# MODIFIED: create_conversational_chain now accepts llm and retriever directly
def create_conversational_chain(llm: BaseChatModel, retriever: VectorStoreRetriever):
    logging.info("Creating conversational chain with message history.")
    
    # Create the base QA chain that handles retrieval and combining documents
    # This chain will be used by RunnableWithMessageHistory
    base_qa_chain = create_qa_chain(llm, retriever) # Pass llm and retriever here

    # Wrap the base_qa_chain with RunnableWithMessageHistory
    return RunnableWithMessageHistory(
        base_qa_chain,
        get_session_history,
        input_messages_key="question", # Key for the user's question in the input dict
        history_messages_key="chat_history" # Key for chat history in the input dict
    )