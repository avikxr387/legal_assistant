# =========================
# IMPORTS
# =========================
# =========================
# IMPORTS
# =========================
from graph.graph_builder import build_graph

# FORCE fallback for now (no OpenAI issues)
RAGAS_AVAILABLE = False

# Optional (keep for later use)
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision
except:
    pass


# =========================
# BUILD GRAPH
# =========================
app = build_graph()


# =========================
# QA PAIRS (GROUND TRUTH)
# =========================
qa_pairs = [
    {
        "question": "What is breach of contract?",
        "ground_truth": "A breach of contract occurs when one party fails to fulfill obligations under a legally binding agreement."
    },
    {
        "question": "What are damages in contract law?",
        "ground_truth": "Damages are monetary compensation awarded for loss caused by breach of contract."
    },
    {
        "question": "What is a termination clause?",
        "ground_truth": "A termination clause defines how and when a contract may be ended."
    },
    {
        "question": "What is force majeure?",
        "ground_truth": "Force majeure refers to unforeseen events that prevent contract performance."
    },
    {
        "question": "What is consideration in contract law?",
        "ground_truth": "Consideration is something of value exchanged between parties in a contract."
    }
]


# =========================
# RUN AGENT
# =========================
def run_agent(question):
    state = {
        "question": question,
        "messages": [],
        "route": "",
        "retrieved": "",
        "sources": [],
        "tool_result": None,
        "answer": "",
        "faithfulness": 0.0,
        "eval_retries": 0
    }

    result = app.invoke(
        state,
        config={"configurable": {"thread_id": "ragas"}}
    )

    return result


# =========================
# COLLECT DATA
# =========================
data = []

for item in qa_pairs:
    result = run_agent(item["question"])

    data.append({
        "question": item["question"],
        "answer": result["answer"],
        "contexts": [result["retrieved"]],
        "ground_truth": item["ground_truth"]
    })


# =========================
# RAGAS EVALUATION
# =========================
if RAGAS_AVAILABLE:
    print("\nRunning RAGAS evaluation...\n")

    dataset = Dataset.from_list(data)

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )

    print(scores)

else:
    print("\nRAGAS not installed. Running fallback evaluation...\n")

    def simple_eval(answer, ground_truth):
        answer = answer.lower()
        gt_words = ground_truth.lower().split()

        matches = sum(1 for word in gt_words if word in answer)

        return round(matches / len(gt_words), 2)

    for item, d in zip(qa_pairs, data):
        score = simple_eval(d["answer"], item["ground_truth"])

        print("\nQ:", item["question"])
        print("Score:", score)