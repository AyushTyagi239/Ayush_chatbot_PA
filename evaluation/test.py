import json
from pathlib import Path
from pydantic import BaseModel, Field

# Path to tests.jsonl inside the same folder as this file
TEST_FILE = Path(__file__).parent / "tests.jsonl"


class TestQuestion(BaseModel):
    """
    A test question used for retrieval and answer evaluation.
    """

    question: str = Field(description="The question to evaluate")
    keywords: list[str] = Field(description="Keywords expected in retrieved context")
    reference_answer: str = Field(description="The correct reference answer")
    category: str = Field(description="Question type: direct_fact, spanning, temporal, etc.")


def load_tests(filepath: str | Path = TEST_FILE) -> list[TestQuestion]:
    """
    Load RAG test questions from a JSONL file.

    Args:
        filepath (str | Path): Path to tests.jsonl

    Returns:
        list[TestQuestion]: List of parsed test cases
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Test file not found: {filepath}")

    tests = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # skip blank lines
                data = json.loads(line.strip())
                tests.append(TestQuestion(**data))

    return tests
