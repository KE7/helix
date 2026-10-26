"""HELIX evaluator for optimize_anything's circle-packing experiment.

The candidate is the experiment's two components: ``solve.py`` is ``code``
and ``refiner_prompt.txt`` is ``refiner_prompt``.  No other file is read.
They are scored by GEPA's own fitness function (``gepa_fitness.py``), and the
result is printed as one ``HELIX_RESULT=`` pair per id in ``helix_batch.json``.

The refiner calls ``openai/gpt-5.1``, so ``OPENAI_API_KEY`` must be set.  The
run's ``StateTracker`` is kept in ``.helix/gepa_state_tracker/`` (or
``$HELIX_CP_STATE_DIR``), with GEPA's ``state_tracker_logs.json`` beside it.
"""

import json
import os
from pathlib import Path

import gepa_fitness


def main() -> None:
    ids = json.loads(Path("helix_batch.json").read_text())
    candidate = {
        "code": Path("solve.py").read_text(),
        "refiner_prompt": Path("refiner_prompt.txt").read_text(),
    }
    refiner = gepa_fitness.make_refiner(api_key=os.environ["OPENAI_API_KEY"])
    with gepa_fitness.StateTracker.open(gepa_fitness.state_dir()) as tracker:
        fitness_fn = gepa_fitness.create_fitness_function(state_tracker=tracker, refiner=refiner)
        score, _output, side_info = fitness_fn(candidate)
    pair = [score, gepa_fitness.json_safe(side_info)]
    print("HELIX_RESULT=" + json.dumps([pair for _ in ids], allow_nan=False))


if __name__ == "__main__":
    main()
