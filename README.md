# 💳 BFSI Assistive Agent (Gemini & LangChain)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework: LangChain](https://img.shields.io/badge/LangChain-0.1.0+-success.svg)](https://www.langchain.com/)
[![Model: Gemini 2.5 Flash](https://img.shields.io/badge/Model-Gemini%202.5%20Flash-blue)](https://ai.google.dev/models/gemini)

This project demonstrates a production-grade **Agentic AI** system in the Banking, Financial Services, and Insurance (BFSI) domain. It simulates a Premium Customer Service and Risk Assessment Agent, showcasing how a Large Language Model (LLM) can achieve **autonomous action** by integrating knowledge retrieval and external tool execution.

---

## 💡 Agentic AI Concepts Explained

The core purpose of this application is to demonstrate the shift from simple Q&A bots to intelligent, autonomous agents.

| Concept | Explanation | Demonstrated In App |
|---------|-------------|---------------------|
| **Agentic RAG** | The Agent dynamically decides *when* to search the internal knowledge base (`card_benefits.txt`) to **ground its answers** in policy, preventing hallucinations. The RAG system acts as a specialized tool, not a fixed step. | **Test RAG-A** and **Test REFLECTION-A** |
| **Autonomous Action** | The Agent is empowered to call external systems (simulated by the `rewards_transfer_api`) to execute real-world, multi-step tasks like initiating a funds transfer. | **Test ACTION-A** |
| **Reflection & Self-Correction** | The Agent observes a failure (e.g., an API returns a risk flag). It then **reflects** on the failure and calls a **second RAG search** (for Fraud Pattern policies) to diagnose and explain the issue to the user. | **Test REFLECTION-A** |

---

## 🛠️ Project Structure and Setup

### Project Files

```text
amex-agentic-concierge-prod/
├── app.py                      # Main Streamlit application, Agent logic, and Tool definitions
├── requirements.txt            # Python dependencies list
├── .env                        # Environment variables (API Key)
├── .gitignore                  # Prevents committing sensitive files
└── data/
    └── card_benefits.txt       # The policy document (RAG knowledge base source)
```

---

### Setup Environment

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/amex-agentic-concierge-prod.git
    cd amex-agentic-concierge-prod
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate 
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configure API Key

1.  **Obtain Key:** Get your **GEMINI_API_KEY** from Google AI Studio.

2.  **Create `.env`:** Create a file named `.env` in the root directory and add your key:
    ```env
    # GEMINI_API_KEY is required for the LLM and Embeddings
    GEMINI_API_KEY="AIzaSy...YOUR_FULL_KEY_HERE"
    ```

### Data and Run

1.  **Data Check:** Ensure the `data/` directory exists and contains the `card_benefits.txt` file (this is the RAG source).

2.  **Launch Application:**
    ```bash
    streamlit run app.py
    ```

---

## 🔍 How to Monitor Agent Reasoning (Logs)

When the app is running, watch the terminal where you executed `streamlit run app.py`. The verbose logging will clearly display the Agent's thought process:

| Log Marker | Agent's Role | Concept Explained |
|------------|--------------|-------------------|
| **`[chain/start]`** | Receives the full customer query. | **Query** |
| **`Thought:`** | **Gemini LLM** reasons about the intent and chooses the next tool. | **Planning/Tool Selection** |
| **`Tool Call:`** | Calls the selected Python function (`policy_search_rag` or `rewards_transfer_api`). | **Action** |
| **`Observation:`** | The output returned by the tool (e.g., RAG policy text or API status). | **Output/Grounding** |
| **`Final Answer:`** | The LLM synthesizes the observation into a final, human-friendly response. | **Outcome** |

---

## 🔬 Testing the Agentic Concepts (Test Plan)

Use the following test cases to interact with the Streamlit app. Select the appropriate **Account ID** in the sidebar to observe the different Agentic concepts in action.

| Test ID | Concept Tested | Account ID | User Query | Expected Terminal Log Behavior |
|---------|----------------|------------|------------|--------------------------------|
| **RAG-A** | **Pure Agentic RAG** (Knowledge Retrieval) | CUST001 | "What is the fee for lounge guests?" | **Must Call:** `policy_search_rag`. **Must NOT Call:** Any action API. |
| **ACTION-A** | **Autonomous Action Success** | **CUST001** | "Transfer 10k points to Delta now." | **Must Call:** `rewards_transfer_api`. **Observation:** Should return `SUCCESS` JSON. |
| **REFLECTION-A** | **Agentic RAG Reflection/Self-Correction** | **CUST002 (Flagged)** | "Transfer 10k points to Delta now." | 1. **Action 1:** Calls `rewards_transfer_api`. 2. **Observation:** Returns `FAILURE`. 3. **Action 2 (Self-Correction):** Calls **`policy_search_rag`** (for Fraud Patterns) to diagnose the failure. |
| **RAG-B** | **Dynamic Tool Chaining** (RAG + Calculator) | CUST003 | "Calculate the fee for transferring 10,000 points." | 1. **Action 1:** Calls `policy_search_rag` to get the fee rate. 2. **Action 2:** Calls **`calculator`** with the resulting formula. |
| **ORCH-A** | **Tool Isolation** | CUST001 | "Tell me about travel insurance." | **Must Call:** `policy_search_rag`. **Must NOT Call:** any other tool. The Agent ignores non-relevant tools based on its reasoning. |

---

## 📊 Key Features

- **Agentic RAG**: Dynamic knowledge retrieval from policy documents
- **Autonomous Actions**: Real-world API integrations for executing customer requests
- **Self-Correction**: Intelligent error handling and reflection mechanisms
- **Multi-Tool Orchestration**: Seamless coordination between RAG, APIs, and calculators
- **Production-Ready Architecture**: Built with LangChain and Gemini 2.5 Flash

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.