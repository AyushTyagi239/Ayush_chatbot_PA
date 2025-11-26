import sys
import math
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import litellm

from evaluation.test import TestQuestion, load_tests
from implementation.answer import answer_question, fetch_context


# -----------------------------
# ENV + NVIDIA QWEN CONFIG
# -----------------------------
load_dotenv(override=True)

litellm.api_key = os.getenv("NVIDIA_API_KEY")
litellm.api_base = "https://integrate.api.nvidia.com/v1"

# Qwen model you requested:
MODEL = "qwen/qwen3-next-80b-a3b-instruct"

db_name = "vector_db"


# -----------------------------
# STRUCTURED OUTPUT MODELS
# -----------------------------

class RetrievalEval(BaseModel):
    mrr: float = Field(description="Mean Reciprocal Rank")
    ndcg: float = Field(description="nDCG score")
    keywords_found: int = Field(description="Keywords located in top-k docs")
    total_keywords: int = Field(description="Total keywords")
    keyword_coverage: float = Field(description="Keywords found %")


class AnswerEval(BaseModel):
    feedback: str = Field(description="Evaluator feedback summary")
    accuracy: float = Field(description="1-5 factual correctness")
    completeness: float = Field(description="1-5 coverage of reference answer")
    relevance: float = Field(description="1-5 relevance to question only")


# -----------------------------
# RETRIEVAL METRICS
# -----------------------------

def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    kw = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, 1):
        if kw in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    return sum(relevances[i] / math.log2(i + 2)
               for i in range(min(k, len(relevances))))


def calculate_ndcg(keyword: str, docs: list, k: int = 10) -> float:
    kw = keyword.lower()

    relevances = [
        1 if kw in doc.page_content.lower() else 0
        for doc in docs[:k]
    ]

    dcg = calculate_dcg(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal, k)

    return (dcg / idcg) if idcg > 0 else 0.0


# -----------------------------
# RETRIEVAL EVALUATION
# -----------------------------

def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    retrieved_docs = fetch_context(test.question)

    mrr_scores = [calculate_mrr(kw, retrieved_docs) for kw in test.keywords]
    ndcg_scores = [calculate_ndcg(kw, retrieved_docs, k) for kw in test.keywords]

    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total = len(test.keywords)

    return RetrievalEval(
        mrr=sum(mrr_scores) / total if total else 0.0,
        ndcg=sum(ndcg_scores) / total if total else 0.0,
        keywords_found=keywords_found,
        total_keywords=total,
        keyword_coverage=100 * keywords_found / total if total else 0.0,
    )


# -----------------------------
# ANSWER QUALITY EVALUATION (LLM JUDGE USING QWEN)
# -----------------------------

def evaluate_answer(test: TestQuestion):
    generated_answer, retrieved_docs = answer_question(test.question)

    judge_messages = [
        {
            "role": "system",
            "content": (
                "You are a strict expert evaluator. Compare the generated answer "
                "with the reference answer and score accurately. Only give 5/5 "
                "when the answer is PERFECT."
            ),
        },
        {
            "role": "user",
            "content": f"""
Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Evaluate the generated answer on:
1. Accuracy (1-5)
2. Completeness (1-5)
3. Relevance (1-5)
Then provide feedback.

Return ONLY valid JSON for AnswerEval.
"""
        }
    ]

    judge_response = litellm.completion(
        model=MODEL,
        messages=judge_messages,
        response_format=AnswerEval
    )

    evaluation = AnswerEval.model_validate_json(
        judge_response.choices[0].message.content
    )

    return evaluation, generated_answer, retrieved_docs


# -----------------------------
# BATCH HELPERS
# -----------------------------

def evaluate_all_retrieval():
    tests = load_tests()
    total = len(tests)
    for idx, test in enumerate(tests):
        yield test, evaluate_retrieval(test), (idx + 1) / total


def evaluate_all_answers():
    tests = load_tests()
    total = len(tests)
    for idx, test in enumerate(tests):
        yield test, evaluate_answer(test)[0], (idx + 1) / total


# -----------------------------
# CLI EXECUTION
# -----------------------------

def run_cli_evaluation(test_number: int):
    tests = load_tests("tests.jsonl")

    if test_number < 0 or test_number >= len(tests):
        print(f"Invalid test index: 0 to {len(tests) - 1}")
        sys.exit(1)

    test = tests[test_number]

    print("\n" + "=" * 80)
    print(f"Test #{test_number}")
    print("=" * 80)
    print("Question:", test.question)
    print("Keywords:", test.keywords)
    print("Reference Answer:", test.reference_answer)

    print("\nRetrieval Evaluation\n" + "=" * 80)
    retrieval_result = evaluate_retrieval(test)
    print(retrieval_result)

    print("\nAnswer Evaluation\n" + "=" * 80)
    answer_result, generated, docs = evaluate_answer(test)
    print("Generated Answer:\n", generated)
    print("\nFeedback:\n", answer_result.feedback)
    print(answer_result)


def main():
    if len(sys.argv) != 2:
        print("Usage: python eval.py <test_number>")
        sys.exit(1)

    test_number = int(sys.argv[1])
    run_cli_evaluation(test_number)


if __name__ == "__main__":
    main()
