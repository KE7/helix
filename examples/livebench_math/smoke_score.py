"""One-row official-scorer gate used after building the evaluator image."""

import json
from pathlib import Path

from constants import LIVEBENCH_CODE_COMMIT, LIVEBENCH_DATA_REVISION, SMOKE_IDS
from scoring import score_livebench_math

data = json.loads(Path("/opt/livebench-math/data.json").read_text())
row = next(
    item
    for item in data["smoke_splits"]["val"]
    if item["question_id"] == SMOKE_IDS["val"][0]
)
correct = score_livebench_math(row, str(row["ground_truth"]))
incorrect = score_livebench_math(row, "definitely-not-the-pinned-answer")
assert correct == 1.0, correct
assert incorrect == 0.0, incorrect
print(
    json.dumps(
        {
            "livebench_code_commit": LIVEBENCH_CODE_COMMIT,
            "dataset_revision": LIVEBENCH_DATA_REVISION,
            "question_id": row["question_id"],
            "subtask": row["subtask"],
            "correct_score": correct,
            "incorrect_score": incorrect,
        },
        sort_keys=True,
    )
)
