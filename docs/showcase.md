# Showcase

Papers, projects and examples built with HELIX. Every number on this page comes from a public paper or from a recorded run in this repository.

Using HELIX in a paper or project? [Open a pull request](https://github.com/KE7/helix/pulls) to add it here.

## Papers

### RHO: 99.0% on Robosuite with a frozen, single-turn robot repository

*RHO: Your Coding Agent is Secretly a Roboticist.* Elmaaroufi, Kalade, Wu, Svegliato, Schelle, Mehr, Seshia and Zaharia. Accepted at the [Sim2Real2Field Workshop @ CoRL 2026](https://sim2real2field-corl26.github.io/) and the [2nd Workshop on Lifelong Agents @ COLM 2026](https://lifelongagent.github.io/).

RHO uses HELIX to evolve multi-file robot-policy repositories in simulation, then ships the result as fixed code, with no LLM calls in the control loop.

**Key results** (from the [project page](https://rho-robotics.github.io/)):

- 99.0% single-turn success on CaP-Bench Robosuite (693/700, GPT-6.1 Sol mutator, best of two searches, on evaluation seeds never seen in search), against CaP-Agent0's 68.29% with multi-turn correction.
- 9.31 s per trial against CaP-Agent0's 113.6 s, 12.2× faster.
- The best search reached 99.0% by generation 17, in 16.5 h, for $20.50 of mutator tokens.

[Paper](https://arxiv.org/abs/2606.16458) · [Project page](https://rho-robotics.github.io/)

## Examples

### ManiSkill pick and place: 0.064 → 1.0 in one mutation

HELIX evolves a multi-file `solver/` package that drives a Franka Panda through push, pick and stack in ManiSkill 3, using four primitives for observation, motion and the gripper. The mutation agent works in HELIX's Docker sandbox and each solver runs in a throwaway container. Contributed by Mahanth Komuravelli.

**Key results** (from the recorded run):

- Validation went from 0.064 (seed) to 1.0 in generation 1, with one accepted mutation.
- The evolved package solves all 141 train, validation and held-out trials. The 90 held-out trials are never shown to the agent.
- The agent session took 6 turns, 5 tool calls and 40 s, and cost $0.14 as reported by Claude Code.

[Write-up](experiments/maniskill-pick-place.md) · [Example](https://github.com/KE7/helix/tree/main/examples/maniskill_pick_place)

### Circle packing

HELIX runs optimize_anything's circle-packing experiment: pack 26 circles in the unit square to maximize the sum of radii, with GEPA's own fitness function and refiner, evolving GEPA's seed code and refiner prompt.

[Write-up](experiments/circle-packing.md) · [Example](https://github.com/KE7/helix/tree/main/examples/circle_packing)

### Web researcher

A question-answering agent for software-engineering questions whose answers change over time, such as current package versions. The seed asks Claude Haiku directly, with no web access, and the evaluator scores answers against versions it fetches live from PyPI and endoflife.date.

[Example](https://github.com/KE7/helix/tree/main/examples/web_researcher)
