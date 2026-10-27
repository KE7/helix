"""
Seed agent for the web_researcher example.

solve(question) sends the question to Claude Haiku and returns the model's
reply. It has no web access, so its answers come from the model's training
data.
"""

import sys

import anthropic

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are a concise software engineering assistant. "
    "Answer with only the requested value, such as a version string, an HTTP "
    "status code, a number, or a module name, with no explanation."
)


def solve(question: str) -> str:
    """Answer a software engineering question by asking the model directly.

    Args:
        question: A natural language question about software.

    Returns:
        The model's answer as a string.
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    text = next((block.text for block in response.content if block.type == "text"), "")
    return text.strip().strip("\"'`").strip()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        "What is the latest stable version of the 'requests' Python package "
        "on PyPI? Answer with just the version number."
    )
    print(solve(question))
