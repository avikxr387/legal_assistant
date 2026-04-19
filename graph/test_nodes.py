# =========================
# IMPORTS
# =========================
from graph.nodes import (
    memory_node,
    router_node,
    retrieval_node,
    skip_retrieval_node,
    tool_node,
    answer_node,
    eval_node,
    save_node
)

from rag.vectordb import create_vector_db


# =========================
# DUMMY LLM (for testing)
# =========================
class DummyLLM:
    def invoke(self, prompt):
        print("\n--- LLM PROMPT ---\n")
        print(prompt[:500])  # print partial prompt
        return "This is a test answer"


llm = DummyLLM()


# =========================
# INIT DB + MODEL
# =========================
collection, model = create_vector_db()


# =========================
# INITIAL STATE
# =========================
state = {
    "question": "What happens when a contract is broken?",
    "messages": [],
    "route": "",
    "retrieved": "",
    "sources": [],
    "tool_result": None,
    "answer": "",
    "faithfulness": 0.0,
    "eval_retries": 0
}


# =========================
# STEP-BY-STEP TESTING
# =========================

print("\n===== MEMORY NODE =====")
state = memory_node(state)
print(state["messages"])


print("\n===== ROUTER NODE =====")
state = router_node(state)
print("Route:", state["route"])


print("\n===== RETRIEVAL NODE =====")
if state["route"] == "retrieve":
    state = retrieval_node(state, collection, model)
else:
    state = skip_retrieval_node(state)

print("Sources:", state["sources"])
print("Context Preview:", state["retrieved"][:200])


print("\n===== TOOL NODE (if needed) =====")
if state["route"] == "tool":
    state = tool_node(state)
    print("Tool Result:", state["tool_result"])


print("\n===== ANSWER NODE =====")
state = answer_node(state, llm)
print("Answer:", state["answer"])


print("\n===== EVAL NODE =====")
state = eval_node(state, llm)
print("Faithfulness:", state["faithfulness"])
print("Retries:", state["eval_retries"])


print("\n===== SAVE NODE =====")
state = save_node(state)
print("Messages:", state["messages"])


print("\n===== FINAL STATE =====")
print(state)