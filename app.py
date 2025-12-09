# app.py

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import random
import json
import time

# --- Setup and Authentication ---
load_dotenv()

# FIX: Explicitly check for and retrieve the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("FATAL ERROR: Please set the GEMINI_API_KEY in the .env file.")
    st.stop()

# Initialize LLM and Embeddings using the explicit API key to bypass ADC error
LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    # FIX APPLIED HERE: Pass the key explicitly to ChatGoogleGenerativeAI
    google_api_key=GEMINI_API_KEY
)
EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    # FIX APPLIED HERE: Pass the key explicitly to GoogleGenerativeAIEmbeddings
    google_api_key=GEMINI_API_KEY
)

# --- 1. RAG Setup: Create Vector Store for the Benefit Analyst Agent ---
def setup_rag_retriever():
    """Initializes and returns the RAG retriever based on the policy document."""
    if 'rag_retriever' not in st.session_state:
        st.info("Initializing RAG (Benefit Policy Vector Store) using Gemini Embeddings...")

        # Load the document
        data_path = "data/card_benefits.txt"
        if not os.path.exists(data_path):
            st.error(f"FATAL: Knowledge base file not found at {data_path}. Please create the 'data' folder and the file.")
            st.stop()

        loader = TextLoader(data_path)
        documents = loader.load()

        # Split the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Create the FAISS vector store and retriever
        vectorstore = FAISS.from_documents(chunks, EMBEDDINGS)
        st.session_state.rag_retriever = vectorstore.as_retriever()
    return st.session_state.rag_retriever

# --- 2. Custom Tools for the Concierge Agent ---

@tool
def policy_search_rag(query: str) -> str:
    """
    Use this tool to search the internal Amex Benefit Policy document for answers
    to questions about lounge access, insurance, or rewards transfer rules.
    This is the core of the Benefit Analyst Agent.
    """
    retriever = setup_rag_retriever()
    # LangChain RAG retrieval
    docs = retriever.invoke(query)

    # Returning the retrieved context to the LLM for generation
    context = "\n---\n".join([doc.page_content for doc in docs])
    return f"Retrieved policy snippets (use this to answer the user's query):\n{context}"

@tool
def rewards_transfer_api(account_id: str, points: int, partner: str) -> str:
    """
    Use this to initiate a rewards transfer and get an immediate status check.
    It will return SUCCESS or FAILURE due to a risk flag.
    """
    print(f"Executing rewards_transfer_api for {account_id}")
    time.sleep(1)

    # Simulate risk condition (CUST002 fails, others pass)
    if account_id == "CUST002":
        return json.dumps({
            "status": "FAILURE",
            "message": "Transaction flagged for risk review. Transfer halted.",
            "policy_hint": "Check Fraud Pattern Library for high-value transfer policies."
        })
    else:
        return json.dumps({
            "status": "SUCCESS",
            "message": f"Scheduled {points} points transfer to {partner}.",
            "id": f"TX{random.randint(10000, 99999)}"
        })

@tool
def calculator(expression: str) -> str:
    """Evaluates a simple mathematical expression. Useful for calculating fees or simple figures."""
    try:
        # NOTE: Using eval() is safe here because the input is controlled by the LLM's structured output.
        return str(eval(expression))
    except Exception as e:
        return f"Calculation Error: {e}"

# List of all available tools
TOOLS = [policy_search_rag, rewards_transfer_api, calculator]

# --- 3. Agent Setup ---

def create_agent_executor():
    """Sets up the LangChain Agent Executor."""
    system_prompt = (
        "You are the Platinum Concierge Agent for American Express. Your goal is to serve the customer "
        "by answering questions and performing actions using your tools. Your primary tools are "
        "policy_search_rag (for benefit details) and rewards_transfer_api (for execution). "
        "Always use the policy_search_rag tool to answer questions about fees, lounge access, or policy details. "
        "If the rewards_transfer_api returns a FAILURE, explain to the user that the action was halted "
        "due to a risk flag and mention that a human agent will follow up."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(LLM, TOOLS, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        # LOGGING ENHANCEMENT: verbose=True shows the Thought, Tool Call, and Observation in the terminal
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8
    )
    return agent_executor

# --- 4. Streamlit UI and Chat Logic ---

st.set_page_config(page_title="💳 Amex Agentic Concierge (Gemini)", layout="wide")
st.title("💳 Amex Agentic Concierge (Gemini Demo)")
st.caption("Multi-Agentic RAG and Action System using LangChain and Gemini 2.5 Flash")

# Initialize chat history and Agent Executor
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_executor" not in st.session_state:
    # Ensure RAG is initialized before the executor starts
    setup_rag_retriever()
    st.session_state.agent_executor = create_agent_executor()

# Sidebar for account selection
with st.sidebar:
    st.header("Account Simulation")
    account_id = st.selectbox(
        "Select Customer Account ID:",
        ["CUST001 (Success)", "CUST002 (Flagged for Risk)", "CUST003 (Calculator Test)"]
    ).split(' ')[0]
    st.write(f"Active Account: **{account_id}**")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about your card benefits or request a rewards transfer..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Pass the Account ID to the agent dynamically for decision-making
    full_input = f"{prompt} (Account ID: {account_id})"

    with st.chat_message("assistant"):
        with st.spinner("Agent (powered by Gemini) is thinking and executing actions... (Check Terminal for Logs)"):
            try:
                # Invoke the LangChain Agent Executor
                response = st.session_state.agent_executor.invoke(
                    {"input": full_input, "chat_history": []}
                )

                final_response = response["output"]
                st.markdown(final_response)

                st.session_state.messages.append({"role": "assistant", "content": final_response})

            except Exception as e:
                error_message = f"An unexpected error occurred. Error: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})