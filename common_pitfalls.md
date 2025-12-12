# Common Pitfalls and How to Avoid Them

Based on building the BFSI Assistant, here are lessons learned:

------------------------------------------------------------------------

## **Pitfall 1: Tool Descriptions Are Too Vague**

### ❌ Bad

``` python
Tool(
    name="search_tool",
    description="Search for information",
    func=search
)
```

### ✅ Good

``` python
Tool(
    name="policy_search_rag",
    description="""Use this ONLY for searching internal policy documents.

Perfect for queries about:
- Card benefits, fees, eligibility
- Reward program rules
- Fraud detection policies

DO NOT use for:
- Executing transfers (use rewards_transfer_api instead)
- Real-time account data (use account_api instead)

Input: Natural language query
Output: Relevant policy text with references""",
    func=policy_search_rag
)
```

------------------------------------------------------------------------

## **Pitfall 2: No Guard Rails on Actions**

### ❌ Bad

``` python
def transfer_api(customer_id, partner, points):
    # Just execute without checks
    return api.transfer(customer_id, partner, points)
```

### ✅ Good

``` python
def transfer_api(customer_id, partner, points):
    # Validate inputs
    if points < 0 or points > 1000000:
        return {"status": "INVALID_AMOUNT"}

    if partner not in VALID_PARTNERS:
        return {"status": "INVALID_PARTNER"}

    # Check account status
    account = get_account(customer_id)
    if account["status"] != "ACTIVE":
        return {"status": "ACCOUNT_INACTIVE"}

    # Check sufficient balance
    if account["reward_balance"] < points:
        return {"status": "INSUFFICIENT_POINTS"}

    # Now execute
    return api.transfer(customer_id, partner, points)
```

------------------------------------------------------------------------

## **Pitfall 3: RAG Returns Too Much Context**

### ❌ Bad

``` python
# Returns 10 pages of policy documents
results = vector_db.search(query, k=50)  # Too many results!
return "\n".join([doc.content for doc in results])
```

### ✅ Good

``` python
# Return focused, relevant excerpts
results = vector_db.search(query, k=3)  # Top 3 only

# Add context about where it came from
formatted_results = []
for doc in results:
    formatted_results.append(f"""
    Source: {doc.metadata['policy_name']}
    Section: {doc.metadata['section']}
    Last Updated: {doc.metadata['updated_date']}

    Content:
    {doc.content[:500]}  # Truncate if too long
    """)

return "\n---\n".join(formatted_results)
```

------------------------------------------------------------------------

## **Pitfall 4: No Timeout Protection**

### ❌ Bad

``` python
# Agent can loop forever
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
    # No max_iterations!
)
```

### ✅ Good

``` python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,          # Stop after 10 steps
    max_execution_time=30,      # Stop after 30 seconds
    early_stopping_method="generate"  # Generate final answer if stuck
)
```
