---
hide:
  - navigation
---

![HELIX](assets/logo/helix-wordmark-light.svg#only-light){ width="320" }
![HELIX](assets/logo/helix-wordmark-dark.svg#only-dark){ width="320" }

# Evolve whole codebases with coding agents

HELIX improves a whole repository against an evaluator you write. Each mutation is a session of a coding agent, such as Claude Code or Codex, that reads, edits, runs and tests the code in its own git worktree before HELIX scores the result. HELIX keeps every candidate that is the best on at least one of your test cases, so a gain on one case is not thrown away for a loss on another. The search loop comes from GEPA's [optimize_anything](https://doi.org/10.1145/3786335.3813167).

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary } [See the results](showcase.md){ .md-button }

<div class="helix-demo" markdown>

| | Seed | Evolved by HELIX |
|---|---|---|
| **Push** | <video src="assets/maniskill/seed-pushcube.mp4" autoplay loop muted playsinline></video> | <video src="assets/maniskill/evolved-pushcube.mp4" autoplay loop muted playsinline></video> |
| **Pick** | <video src="assets/maniskill/seed-pickcube.mp4" autoplay loop muted playsinline></video> | <video src="assets/maniskill/evolved-pickcube.mp4" autoplay loop muted playsinline></video> |
| **Stack** | <video src="assets/maniskill/seed-stackcube.mp4" autoplay loop muted playsinline></video> | <video src="assets/maniskill/evolved-stackcube.mp4" autoplay loop muted playsinline></video> |

</div>

On ManiSkill 3, one HELIX mutation by Claude Sonnet 5.5 rewrote a Franka Panda solver package. Validation went from 0.064 to 1.0, and the evolved solver succeeds on all 141 train, validation and held-out trials. The videos show validation trial 26 of each task, with the wrist camera in the top left. [Read the write-up](experiments/maniskill-pick-place.md).

## Watch the search live

`helix ui` shows a run in your browser, while it runs or after it ends. You can follow each agent's transcript as it works, see which proposals were kept and which were rejected, and open any candidate to read its diff and its score on each instance. It never writes to the run. [Read the guide](guides/ui.md).

<div class="helix-ui" markdown>
<video class="ui-light" src="assets/ui/helix-ui-tour-light.mp4" poster="assets/ui/helix-ui-light.png" aria-label="helix ui on a live eterna100 RNA design run" width="1440" height="900" autoplay loop muted playsinline></video>
<video class="ui-dark" src="assets/ui/helix-ui-tour-dark.mp4" poster="assets/ui/helix-ui-dark.png" aria-label="helix ui on a live eterna100 RNA design run" width="1440" height="900" autoplay loop muted playsinline></video>
</div>

## Results

| | |
|---|---|
| **0.064 → 1.0** | ManiSkill push, pick and stack: one Claude Sonnet 5.5 mutation (40 s of agent time, $0.14) solves all 141 trials, including 90 held out. [Write-up](experiments/maniskill-pick-place.md) |
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
  .helix-ui video { width: 100%; height: auto; display: block; border-radius: 6px; border: 1px solid var(--md-default-fg-color--lightest); }
  [data-md-color-scheme="slate"] .helix-ui .ui-light, [data-md-color-scheme="default"] .helix-ui .ui-dark { display: none; }
</style>
