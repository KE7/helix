# License and attribution

This demo does not vendor a FormulaCode dataset row, benchmark clone, Docker
image, solution patch, or generated result. It downloads or measures them only
inside the ignored `examples/formulacode/.work/` directory.

- **FormulaCode / fc-eval** — copyright 2026 FormulaCode Developers,
  BSD-3-Clause. Scoring formulas and failure semantics are derived from
  `adapters/formulacode/template/parser.py` and `docs/guides/metrics.md` at
  commit `c08f665e7bf3b4de225b72dc02ce9b15b7aaba2b`.
- **FormulaCode verified dataset** — pinned in `pins.json`. The demo reads only
  task metadata during source verification; the task's oracle patch is never
  copied into the mutation workspace.
- **NetworkX** — copyright NetworkX Developers, BSD-3-Clause. The smoke task
  checks out base `a986762f2a1919126df2174644232c92c58be2be`; the human oracle is
  measured from merge `3d0bb212f9fa4bac168c3b8c3f512a5f69b7920c` in a temporary,
  mutation-inaccessible reference worktree.
- **Official workload source** — NetworkX pull request #7971. The graph shape
  is the PR author's published singleton-plus-complete-component benchmark.

FormulaCode citation:

> Atharva Sehgal, James Hou, Akanksha Sarkar, Ishaan Mantripragada, Swarat
> Chaudhuri, Jennifer J. Sun, and Yisong Yue. “Evaluating Agentic Optimization
> on Large Codebases.” arXiv:2603.16011, 2026.
