# 💳 BFSI Agentic Assistant (Gemini & LangChain)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework: LangChain](https://img.shields.io/badge/LangChain-0.1.0+-success.svg)](https://www.langchain.com/)
[![Model: Gemini 2.5 Flash](https://img.shields.io/badge/Model-Gemini%202.5%20Flash-blue)](https://ai.google.dev/models/gemini)

This project demonstrates a working **Agentic AI** system in the Banking, Financial Services, and Insurance (BFSI) domain. It simulates a Premium Customer Service and Risk Assessment Agent, showcasing how a Large Language Model (LLM) can achieve **autonomous action** by integrating knowledge retrieval and external tool execution.

---

## 💡 Agentic AI Concepts Explained

The core purpose of this application is to demonstrate the shift from simple Q&A bots to intelligent, autonomous agents.

| Concept | Explanation | Demonstrated In App |
|---------|-------------|---------------------|
| **Agentic RAG** | The Agent dynamically decides *when* to search the internal knowledge base (`card_benefits.txt`) to **ground its answers** in policy, preventing hallucinations. The RAG system acts as a specialized tool, not a fixed step. | **RAG-A**, **REFLECTION-A**, **RAG-B** |
| **Autonomous Action** | The Agent is empowered to call external systems (simulated by the `rewards_transfer_api`) to execute real-world, multi-step tasks like initiating a funds transfer. | **ACTION-A** |
| **Reflection & Self-Correction** | The Agent observes a failure (e.g., an API returns a risk flag). It then **reflects** on the failure and calls a **second RAG search** (for Fraud Pattern policies) to diagnose and explain the issue to the user. | **REFLECTION-A** |
| **Multi-Agent Orchestration** | Multiple specialized agents work together: Document Verification Agent, Credit Analysis Agent, and Decision Agent coordinate to process complex loan applications. | **MULTI-A**, **MULTI-B** |
| **Tool Chaining** | The Agent chains multiple tools together in sequence (RAG → Calculator) to solve complex queries requiring multiple steps. | **RAG-B** |

---

## 🛠️ Project Structure

```text
bfsi-agentic-assistant/
├── app.py                      # Main Streamlit application with Agent logic
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API Key)
├── .gitignore                  # Prevents committing sensitive files
└── data/
    ├── card_benefits.txt       # Card card policy document (RAG source)
    └── loan_policies.txt       # Loan eligibility policies (Multi-Agent RAG)
```

---

## 🚀 Quick Start Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/bfsi-agentic-assistant.git
cd bfsi-agentic-assistant
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
```
streamlit
python-dotenv
langchain
langchain-community
langchain-google-genai
langchain-huggingface
sentence-transformers
faiss-cpu
```

### Step 4: Configure API Key

1. **Get your Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)

2. **Create `.env` file** in the project root:

```env
GEMINI_API_KEY="AIzaSy...YOUR_FULL_KEY_HERE"
```

> **Note:** The `.env` file should never be committed to Git (it's in `.gitignore`)

### Step 5: Prepare Data Files

Ensure the `data/` directory contains both required files:

**`data/card_benefits.txt`** - Sample content:
```text
Platinum Card Benefits:

Annual Fee: $695

Lounge Access:
- Complimentary access to 1,400+ airport lounges worldwide
- Guest fee: $50 per guest per visit
- Priority Pass Select membership included

Travel Insurance:
- Trip cancellation coverage up to $10,000 per trip
- Baggage insurance up to $3,000
- Car rental loss and damage coverage

Points Transfer:
- Transfer points 1:1 to airline and hotel partners
- Federal excise tax offset fee: $0.0006 per point applies to airline transfers
- Partners include Delta, Marriott, Hilton, and more

Fraud Pattern Policy:
- Transfers flagged for unusual volume or destination
- Manual review required for high-risk transactions
- Error code HIGH_RISK_FLAG_007 indicates fraud review needed
```

**`data/loan_policies.txt`** - Sample content:
```text
Loan Eligibility Criteria:

Minimum Credit Score: 720
Maximum Debt-to-Income Ratio: 0.40 (40%)
Maximum Loan Amount: $100,000
Standard Interest Rate: 5.99% APR

Document Requirements:
- Government-issued photo ID
- Proof of income (last 3 pay stubs)
- Bank statements (last 3 months)
- Employment verification

Approval Process:
1. Document verification and fraud check
2. Credit bureau score verification
3. Income and DTI ratio validation
4. Final underwriting decision
```

### Step 6: Launch the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 🔍 Understanding the Interface

### Main Components

1. **Left Sidebar**: Test case selector to simulate different scenarios
2. **Chat Input**: Ask questions or make requests
3. **Terminal/Console**: View detailed agent reasoning logs (verbose mode)

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        BFSI Agentic Assistant                             │
│                         (Streamlit Web Interface)                         │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          AGENT ORCHESTRATION LAYER                        │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │              Gemini 2.5 Flash Lite (LLM Core)                  │     │
│  │  • Reasoning & Planning                                        │     │
│  │  • Tool Selection Logic                                        │     │
│  │  • Response Synthesis                                          │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                │                                          │
│                                ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │           LangChain Agent Executor (Controller)                │     │
│  │  • Query Routing                                               │     │
│  │  • Multi-Step Planning                                         │     │
│  │  • Tool Invocation Management                                  │     │
│  └────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
┌─────────────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│     TOOL 1: RAG         │  │  TOOL 2: APIs   │  │ TOOL 3: COMPUTE  │
│   (Knowledge Base)      │  │  (Actions)      │  │  (Calculator)    │
├─────────────────────────┤  ├─────────────────┤  ├──────────────────┤
│                         │  │                 │  │                  │
│ ┌─────────────────────┐ │  │ • Rewards      │  │ • Math Parser    │
│ │ policy_search_rag   │ │  │   Transfer API │  │ • Expression     │
│ └─────────────────────┘ │  │ • Decision     │  │   Evaluator      │
│           │             │  │   Agent        │  │                  │
│           ▼             │  │   (Multi-Agent)│  │                  │
│ ┌─────────────────────┐ │  └─────────────────┘  └──────────────────┘
│ │   Query Embedding   │ │
│ │   (HuggingFace)     │ │
│ │   • all-MiniLM-L6   │ │
│ │   • Local/Offline   │ │
│ │   • 384-dim vectors │ │
│ └─────────────────────┘ │
│           │             │
│           ▼             │
│ ┌─────────────────────┐ │
│ │  FAISS Vector DB    │ │
│ │  (In-Memory Index)  │ │
│ └─────────────────────┘ │
│           │             │
│           ▼             │
│ ┌─────────────────────┐ │
│ │ Similarity Search   │ │
│ │ Top-K Retrieval     │ │
│ └─────────────────────┘ │
│           │             │
│           ▼             │
│ ┌─────────────────────┐ │
│ │ Return Policy Docs  │ │
│ └─────────────────────┘ │
└─────────────────────────┘

                    DATA SOURCES (Static Files)
┌────────────────────────────────────────────────────────────────┐
│  • data/card_benefits.txt    (Card Policies - RAG Source 1)   │
│  • data/loan_policies.txt    (Loan Policies - RAG Source 2)   │
└────────────────────────────────────────────────────────────────┘
```

### Data Flow Example: "What is the fee for lounge guests?"

```
1. User Query ───▶ 2. Agent Executor ───▶ 3. Gemini LLM Reasoning
      │                    │                        │
      │                    │                   "Need policy info"
      │                    ▼                        │
      │           4. Select Tool: RAG              │
      │                    │◀───────────────────────┘
      │                    │
      │                    ▼
      │           5. policy_search_rag("lounge guest fee")
      │                    │
      │                    ├──▶ 6. Embed Query (HuggingFace)
      │                    │         "lounge guest fee" → [0.23, -0.15, ...]
      │                    │
      │                    ├──▶ 7. Search FAISS Vector DB
      │                    │         Similarity Search
      │                    │         ↓
      │                    │    8. Retrieve Top-K Docs
      │                    │       "Guest fee: $50 per visit..."
      │                    │
      │                    ▼
      │           9. Return Retrieved Text ───▶ 10. Gemini Synthesis
      │                                               │
      │                                          "Based on policy..."
      │                                               │
      └──────────────────────────────────────────────▼
                                        11. Final Answer to User
                                           "$50 per guest per visit"
```

### Key Technical Components

| Component              | Technology                        | Purpose                             | Location |
|------------------------|-----------------------------------|-------------------------------------|----------|
| **LLM Brain**          | Gemini 2.5 Flash Lite             | Reasoning, planning, tool selection | Cloud API |
| **Agent Orchestrator** | LangChain, Agent Executor         | Routes queries, manages tool calls  | Local |
| **Embeddings**         | HuggingFace Sentence-Transformers | Convert text to vectors (384-dim)   | Local |
| **Vector Store**       | FAISS (Facebook AI)               | Fast similarity search, retrieval   | In-Memory |
| **RAG Source 1**       | card_benefits.txt                 | Card policy knowledge base          | Local File |
| **RAG Source 2**       | loan_policies.txt                 | Loan eligibility policies           | Local File |
| **Action APIs**        | Python Functions                  | Simulate external services          | Local |
| **UI Layer**           | Streamlit                         | Web interface, chat display         | Local |

### Why This Architecture?

**✅ Cost Efficiency:**
- Only Gemini API calls cost money (reasoning/chat)
- Embeddings are free and run locally (no Google API quota)
- Vector DB runs in-memory (no cloud database costs)

**✅ Performance:**
- HuggingFace model loaded once, cached
- FAISS provides sub-millisecond similarity search
- Agent executor caches retrievers for fast access

**✅ Privacy:**
- Policy documents never leave your machine
- Embeddings computed locally
- Only user queries sent to Gemini API

**✅ Scalability:**
- Easy to add more RAG sources (just add .txt files)
- Tool system is modular (add new tools as Python functions)
- FAISS can handle millions of vectors efficiently

---

## 🧪 Test Scenarios Guide

### How to Test

1. **Select a test case** from the sidebar dropdown
2. **Copy the user query** from the table below
3. **Paste it** into the chat input
4. **Watch the terminal** to see agent reasoning in real-time
5. **Review the response** in the chat interface

---

### Test Case 1: RAG-A (Pure Knowledge Retrieval)

**Concept:** Agent retrieves information from policy documents without taking actions

**Setup:**
- **Sidebar Selection:** `RAG-A (Policy Retrieval)`
- **Account ID:** `CUST001`

**User Query:**
```
What is the fee for lounge guests?
```

**Expected Behavior:**
- ✅ Agent calls `policy_search_rag` tool
- ✅ Retrieves policy text about lounge fees
- ✅ Returns: "$50 per guest per visit"
- ❌ Does NOT call any action APIs

**Terminal Log Pattern:**
```
Thought: I need to search the policy documents
Action: policy_search_rag
Action Input: lounge guest fee
Observation: Retrieved policy text... Guest fee: $50 per guest...
Final Answer: The lounge guest fee is $50 per visit.
```

---

### Test Case 2: ACTION-A (Autonomous Action Success)

**Concept:** Agent executes a real action via API call

**Setup:**
- **Sidebar Selection:** `ACTION-A (Action Success)`
- **Account ID:** `CUST001`

**User Query:**
```
Transfer 10k points to Delta now.
```

**Expected Behavior:**
- ✅ Agent extracts: account_id=CUST001, points=10000, partner=Delta
- ✅ Calls `rewards_transfer_api(CUST001, 10000, "Delta")`
- ✅ Returns SUCCESS with transaction ID
- ✅ Confirms transfer completion

**Terminal Log Pattern:**
```
Thought: User wants to transfer points, I'll execute the transfer
Action: rewards_transfer_api
Action Input: {"account_id": "CUST001", "points": 10000, "partner": "Delta"}
Observation: {"status": "SUCCESS", "transaction_id": "TX123456", ...}
Final Answer: Successfully transferred 10,000 points to Delta. Transaction ID: TX123456
```

---

### Test Case 3: REFLECTION-A (Self-Correction)

**Concept:** Agent detects failure, reflects, and searches for explanation

**Setup:**
- **Sidebar Selection:** `REFLECTION-A (Action Failure & RAG)`
- **Account ID:** `CUST002` (flagged account)

**User Query:**
```
Transfer 10k points to Delta now.
```

**Expected Behavior:**
1. ✅ Agent calls `rewards_transfer_api`
2. ✅ API returns FAILURE with error code `HIGH_RISK_FLAG_007`
3. ✅ **Agent reflects** and calls `policy_search_rag` for "fraud pattern policy"
4. ✅ Explains why transfer failed based on policy

**Terminal Log Pattern:**
```
Thought: I'll execute the transfer
Action: rewards_transfer_api
Observation: {"status": "FAILURE", "error_code": "HIGH_RISK_FLAG_007"}

Thought: Transfer failed, I need to understand why
Action: policy_search_rag
Action Input: fraud pattern policy
Observation: Retrieved policy... HIGH_RISK_FLAG_007 indicates fraud review...

Final Answer: The transfer was flagged for review due to unusual activity...
```

---

### Test Case 4: RAG-B (Tool Chaining)

**Concept:** Agent chains RAG → Calculator to solve complex queries

**Setup:**
- **Sidebar Selection:** `RAG-B (Tool Chaining/Calc)`
- **Account ID:** `CUST003`

**User Query:**
```
Calculate the fee for transferring 10,000 points.
```

**Expected Behavior:**
1. ✅ Agent calls `policy_search_rag` to find fee rate ($0.0006/point)
2. ✅ Agent calls `calculator` with expression "10000 * 0.0006"
3. ✅ Returns calculated result: $6.00

**Terminal Log Pattern:**
```
Thought: I need to find the fee rate first
Action: policy_search_rag
Observation: Federal excise tax offset fee: $0.0006 per point...

Thought: Now I'll calculate the total fee
Action: calculator
Action Input: 10000 * 0.0006
Observation: 6.0

Final Answer: The fee for transferring 10,000 points is $6.00
```

---

### Test Case 5: ORCH-A (Tool Isolation)

**Concept:** Agent selects only relevant tools, ignores others

**Setup:**
- **Sidebar Selection:** `ORCH-A (Tool Isolation)`
- **Account ID:** `CUST001`

**User Query:**
```
Tell me about travel insurance.
```

**Expected Behavior:**
- ✅ Agent calls ONLY `policy_search_rag`
- ❌ Does NOT call `rewards_transfer_api` or `calculator`
- ✅ Returns insurance policy details

---

### Test Case 6: MULTI-A (Multi-Agent Success)

**Concept:** Orchestrator coordinates multiple specialized agents

**Setup:**
- **Sidebar Selection:** `MULTI-A (Multi-Agent Success)`
- **Account ID:** `CUST_LOAN_750`

**User Query:**
```
Process loan application APP001 for customer CUST_LOAN_750
```

**Expected Behavior:**
1. ✅ Orchestrator calls `decision_agent`
2. ✅ Decision agent calls `document_verification_agent` → ✅ Verified
3. ✅ Decision agent calls `credit_analysis_agent` → Score: 750, DTI: 0.35
4. ✅ Final decision: **APPROVE** (meets criteria: score > 720, DTI < 0.40)

**Terminal Log Pattern:**
```
Thought: I need to process the loan application
Action: decision_agent
Observation: Processing application...
  - Document verification: PASSED
  - Credit score: 750 (requirement: >720)
  - DTI ratio: 0.35 (requirement: <0.40)
  - Decision: APPROVE
  - Loan amount: $75,000
  - Interest rate: 5.99%
```

---

### Test Case 7: MULTI-B (Multi-Agent Review)

**Concept:** Multi-agent system flags application for manual review

**Setup:**
- **Sidebar Selection:** `MULTI-B (Multi-Agent Review)`
- **Account ID:** `CUST_LOAN_680`

**User Query:**
```
Process loan application APP002 for customer CUST_LOAN_680
```

**Expected Behavior:**
1. ✅ Document verification passes
2. ✅ Credit analysis returns: Score: 680, DTI: 0.50
3. ✅ Decision: **REVIEW** (does not meet auto-approval criteria)
4. ✅ Explains manual review needed

---

## 📊 Key Features

### 🎯 Core Capabilities

- **Agentic RAG**: Dynamic knowledge retrieval from policy documents
- **Autonomous Actions**: Real-world API integrations for executing customer requests
- **Self-Correction**: Intelligent error handling and reflection mechanisms
- **Multi-Tool Orchestration**: Seamless coordination between RAG, APIs, and calculators
- **Multi-Agent Architecture**: Specialized agents working together for complex workflows

### 🔧 Technical Highlights

- **Free Embeddings**: Uses HuggingFace (sentence-transformers) - no API costs
- **No Quota Limits**: Local embeddings eliminate rate limiting issues
- **Fast Performance**: In-memory FAISS vector store for quick retrieval
- **Verbose Logging**: Real-time agent reasoning displayed in terminal
- **Stateful Sessions**: Maintains conversation context across interactions

---

## 🐛 Troubleshooting

### Issue: "GEMINI_API_KEY not found"

**Solution:**
```bash
# Verify .env file exists
ls -la .env

# Check contents (should show your key)
cat .env
```

### Issue: "Unresolved reference 'langchain_huggingface'"

**Solution:**
```bash
# Install correct packages
pip install langchain-community sentence-transformers
```

### Issue: Agent not finding documents

**Solution:**
```bash
# Verify data files exist
ls -la data/

# Should show:
# - card_benefits.txt
# - loan_policies.txt

# Clear cache and restart
rm -rf ~/.streamlit/cache/
streamlit run app.py
```

### Issue: First run is slow

**Explanation:** HuggingFace downloads the embedding model (~90MB) on first run. Subsequent runs are fast.

### Issue: Agent asks for account ID

**Solution:** Make sure you've selected a test case from the sidebar. The account ID is automatically included in the context.

---

## 🔍 Monitoring Agent Reasoning

The terminal where you run `streamlit run app.py` shows detailed logs:

| Log Marker | Meaning |
|------------|---------|
| `[chain/start]` | Agent receives user query |
| `Thought:` | Agent's reasoning process |
| `Action:` | Tool being called |
| `Action Input:` | Arguments passed to tool |
| `Observation:` | Tool's return value |
| `Final Answer:` | Agent's synthesized response |

**Example Log:**
```
> Entering new AgentExecutor chain...

Thought: The user wants to know about lounge fees
Action: policy_search_rag
Action Input: lounge guest fee

Observation: Retrieved policy text:
---
Lounge Access:
- Guest fee: $50 per guest per visit
---

Thought: I now have the information needed
Final Answer: The lounge guest fee is $50 per visit.

> Finished chain.
```

---

## 📈 Performance Optimization

### Caching Strategy

The app uses Streamlit's `@st.cache_resource` to cache:
- RAG retrievers (loaded once)
- Agent executor (created once)
- Embedding model (downloaded once)

### Memory Management

- FAISS index stored in memory for fast retrieval
- HuggingFace model loaded once, reused for all embeddings
- Chat history limited to last 10 messages to prevent context overflow

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FAISS Documentation](https://faiss.ai/)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/) framework
- Powered by [Google Gemini 2.5 Flash](https://ai.google.dev/models/gemini)
- Uses [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers)
- UI built with [Streamlit](https://streamlit.io/)

---

## ⭐ Support

If you find this project useful:
- Give it a **Star ⭐** on GitHub
- **Share** it with others interested in Agentic AI
- **Follow** for updates on new features

---
