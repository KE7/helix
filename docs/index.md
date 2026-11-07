---
hide:
  - navigation
---

![HELIX](assets/logo/helix-wordmark-light.svg#only-light){ width="320" }
![HELIX](assets/logo/helix-wordmark-dark.svg#only-dark){ width="320" }

# Evolve whole codebases with coding agents

HELIX optimizes a whole repository against your evaluator. Each mutation is a multi-turn session of a coding agent (Claude Code, Codex, Cursor, OpenCode or Antigravity) that reads, edits, runs and tests code in its own git worktree before HELIX scores the result. Candidates are kept on an instance-level Pareto frontier, so a program that is best on any one case survives. The search loop comes from GEPA's [optimize_anything](https://doi.org/10.1145/3786335.3813167).

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary } [See the results](showcase.md){ .md-button }

<div class="helix-demo" markdown>

| | Seed | Evolved by HELIX |
|---|---|---|
| **Push** | <video src="assets/maniskill/seed-pushcube.mp4" autoplay loop muted playsinline></video> | <video src="assets/maniskill/evolved-pushcube.mp4" autoplay loop muted playsinline></video> |
| **Pick** | <video src="assets/maniskill/seed-pickcube.mp4" autoplay loop muted playsinline></video> | <video src="assets/maniskill/evolved-pickcube.mp4" autoplay loop muted playsinline></video> |
| **Stack** | <video src="assets/maniskill/seed-stackcube.mp4" autoplay loop muted playsinline></video> | <video src="assets/maniskill/evolved-stackcube.mp4" autoplay loop muted playsinline></video> |

</div>

One HELIX mutation by Claude Sonnet rewrote a Franka Panda solver package on ManiSkill 3: validation went from 0.064 to 1.0, and the evolved package solves all 141 train, validation and held-out trials. The videos show validation trial 26 of each task, with the wrist camera in the top left. [Read the write-up](experiments/maniskill-pick-place.md).

## Results

| | |
|---|---|
| **0.064 → 1.0** | ManiSkill push, pick and stack: one Claude Sonnet mutation (40 s of agent time, $0.14) solves all 141 trials, including 90 held out. [Write-up](experiments/maniskill-pick-place.md) |
| **99.0%** | [RHO](https://rho-robotics.github.io/) on Robosuite: HELIX-evolved robot-policy repositories reach 99.0% single-turn success (693/700), against CaP-Agent0's 68.29% with multi-turn correction. |
| **12.2× faster** | RHO's evolved repositories run as fixed code with no LLM calls in the control loop: 9.31 s per trial, against CaP-Agent0's 113.6 s. |

## How it works

![The HELIX search loop: pick a parent from the instance-level Pareto frontier, let a coding agent edit it over many turns in its own worktree, gate the child against its parent on a training minibatch, score it on the full validation set, and add it to the frontier if it is the best on some instance. Both scoring steps use a protected evaluator.](assets/helix-loop-light.svg#only-light)
![The HELIX search loop](assets/helix-loop-dark.svg#only-dark)

More in [How HELIX works](concepts/how-it-works.md).

<style>
  .helix-demo table { width: 100%; }
  .helix-demo video { width: 100%; max-width: 420px; display: block; border-radius: 4px; }
  .md-typeset td strong { white-space: nowrap; }
</style>
