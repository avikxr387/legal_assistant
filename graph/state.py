from typing import TypedDict, List, Optional


class CapstoneState(TypedDict):
    # 🔹 User input
    question: str

    # 🔹 Conversation memory (for MemorySaver)
    messages: List[dict]

    # 🔹 Routing decision (retrieval / tool / end / retry)
    route: str

    # 🔹 Retrieved documents (RAG output)
    retrieved: List[str]

    # 🔹 Source metadata (topics, doc ids)
    sources: List[str]

    # 🔹 Tool output (if any tool is used)
    tool_result: Optional[str]

    # 🔹 Final generated answer
    answer: str

    # 🔹 Evaluation score (faithfulness)
    faithfulness: float

    # 🔹 Retry counter (for eval loop)
    eval_retries: int
    
    full_doc_text: Optional[str]