# =========================
# IMPORTS
# =========================
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import CapstoneState
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
# INIT DB + MODEL
# =========================
collection, model = create_vector_db()


# =========================
# DUMMY LLM (replace later)
# =========================
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class GroqLLM:
    def invoke(self, prompt):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # fast + good
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


llm = GroqLLM()


# =========================
# WRAPPER FUNCTIONS
# =========================
def retrieval_wrapper(state):
    return retrieval_node(state, collection, model)


def answer_wrapper(state):
    return answer_node(state, llm)


def eval_wrapper(state):
    return eval_node(state, llm)


# =========================
# DECISION FUNCTIONS
# =========================
def route_decision(state: CapstoneState):
    return state["route"]


def eval_decision(state: CapstoneState):
    if state["faithfulness"] < 0.7 and state["eval_retries"] < 2:
        return "answer"
    return "save"


# =========================
# BUILD GRAPH
# =========================
def build_graph():
    graph = StateGraph(CapstoneState)

    # Nodes
    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_wrapper)
    graph.add_node("skip", skip_retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_wrapper)
    graph.add_node("eval", eval_wrapper)
    graph.add_node("save", save_node)

    # Entry
    graph.set_entry_point("memory")

    # Fixed edges
    graph.add_edge("memory", "router")

    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")

    graph.add_edge("answer", "eval")
    graph.add_edge("save", END)

    # Conditional edges (router)
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "skip": "skip",
            "tool": "tool"
        }
    )

    # Conditional edges (eval)
    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {
            "answer": "answer",  # retry
            "save": "save"
        }
    )

    # Compile
    app = graph.compile(checkpointer=MemorySaver())

    print("Graph compiled successfully")

    return app, collection, model