"""Run one circle-packing candidate the way GEPA's subprocess executor does.

Usage: ``python run_main.py <code.py> <best.json> <result.json> <timeout>``

Like the wrapper script of ``gepa.utils.code_execution`` (subprocess mode), it
``exec``s the code in a fresh namespace named ``__main__``, calls
``main(timeout=..., current_best_solution=...)`` and writes the outcome; an
exception is caught and reported, anything that ends the process early leaves
no result file.  It writes JSON instead of a pickle: the circles as
``np.array(circles).tolist()`` and the scores as floats, which is all GEPA's
``execute_code`` reads from the return value.
"""

import json
import sys
import traceback

import numpy as np


def describe(value):
    """The parts of main()'s return value that GEPA's execute_code reads."""
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    out = {"type": "dict", "has_circles": "circles" in value, "has_all_scores": "all_scores" in value}
    if out["has_circles"]:
        try:
            circles = np.array(value["circles"]).tolist()
            json.dumps(circles)
            out["circles"] = circles
        except Exception as e:
            out["circles_error"] = str(e)
    if out["has_all_scores"]:
        try:
            out["all_scores"] = [float(s) for s in value["all_scores"]]
        except Exception:
            out["all_scores"] = None
    return out


def main():
    code_path, best_path, out_path, timeout = sys.argv[1:5]
    with open(code_path) as f:
        code = f.read()
    with open(best_path) as f:
        best = json.load(f)
    current_best_solution = None if best is None else np.array(best)

    context = {"__name__": "__main__"}
    output = {"success": False, "error": "Code did not complete"}
    try:
        exec(code, context)
        returned = None
        if "main" in context:
            returned = context["main"](timeout=int(timeout), current_best_solution=current_best_solution)
        output = {"success": True, "return": describe(returned)}
    except Exception as e:
        output = {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    with open(out_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
