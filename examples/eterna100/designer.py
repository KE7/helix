"""
Seed RNA designer for the eterna100 example.

design(target, budget_s, seed) returns an RNA sequence (A, C, G, U) of the
same length as ``target``, a secondary structure in dot-bracket notation.
The design succeeds when the sequence's minimum-free-energy structure under
ViennaRNA 2 is exactly ``target``.

This seed does no search: every base pair is G-C and every unpaired base is
A. It solves a puzzle only when that pattern happens to fold right.
"""

import sys


def design(target: str, budget_s: float = 8.0, seed: int = 0) -> str:
    """Return a sequence for the dot-bracket ``target``.

    Args:
        target: Secondary structure in dot-bracket notation, e.g. "((((....))))".
        budget_s: Seconds the evaluator allows for this puzzle.
        seed: Seed for any randomness, so a design can be reproduced.

    Returns:
        A string over A, C, G, U with ``len(target)`` letters.
    """
    return "".join("G" if c == "(" else "C" if c == ")" else "A" for c in target)


if __name__ == "__main__":
    print(design(sys.argv[1] if len(sys.argv) > 1 else "((((((....))))))"))
