# UnionMimic

Read [MAINLINE](docs/MAINLINE.md) for the accepted goal and [STATUS](docs/STATUS.md) for evidence. For the task at hand, read [architecture](docs/ARCHITECTURE.md), [experiments](docs/EXPERIMENTS.md), or [training](docs/TRAINING.md).

## Mainline

- Build SuperMimic’s complete online loop on the existing Sharpa hand, then compare SM with SM plus WM-Craftnet recurrent policy context. Keep one object for now; extend actions before object diversity, retaining z-axis rotation as a diagnostic control. Actor/critic start untrained; reuse compatible WSM weights as the prior. Keep SM physical dynamics and WSM context separate.
- Current priority: establish a strong Sharpa SM baseline through reference-code and adaptation audits; follow [SM_REPRODUCTION](docs/SM_REPRODUCTION.md). Fix and tune the shared SM baseline before claiming prior benefits; retain those improvements in both SM and SM+WSM. Use a pinned reference learner/recipe as the starting point; document necessary adaptations and verify equivalence before tuning. Do not promote smoke-test scales or blended defaults to reproduction settings; see [reference audit](docs/REFERENCE_BASELINE_AUDIT_20261103.md).
- Latest order (2026-11-04): complete Sharpa multi-action adaptation first, then compare SM/PPO on that verified interface, then add WM-Craftnet context. Corner-block z-rotation only demonstrates the narrower learning loop. Both single-axis long runs were stopped at this direction; do not resume them. See [adaptation acceptance](docs/SHARPA_FULL_ADAPTATION_20261104.md).
- Policy-context benefit, real/imagined context consistency, and improved dynamics prediction are different hypotheses. Preserve SM’s observation history and context; never use future recorded observations in actor imagination. Frozen weights still permit state updates and input gradients.
- Follow the latest explicit user direction and update MAINLINE when it changes. Historical warm-start runs and humanoid examples do not redefine the goal. Extensions must not displace the complete SM baseline.

- Latest presentation priority (2026-11-09): build the story and demos around reusable contact context in SM imagination before waiting for complete reproduction. Follow [story and demo](docs/STORY_AND_DEMO.md). Keep evidence labels; reference playback is not policy benefit. Background acceptance and the authorized queue may continue.

- Prior feasibility first: freeze WSM parameters; distinguish recurrent state propagation from weight optimization. Check added information, total cost, 10Hz/50Hz timing and incremental/residual action semantics before integration. State propagation alone is not a research contribution; see [measured audit](docs/FROZEN_CONTEXT_AUDIT_20261109.md). See the completed [three-stage study](docs/CONTEXT_STUDY_20261109.md): 100 actor updates did not establish control benefit; offline geometry readout is only an exploratory signal. Do not call it precision-control improvement or start whole-WSM distillation by default.

- Latest approved follow-up (2026-11-09 evening): compact frozen-encoder geometry readout, random-encoder controls, calibrated imagination input, and labeled demo backups; see [geometry study](docs/GEOMETRY_AND_DEMO_BACKUP_20261109.md). Independent-seed position error improved; no policy benefit yet. The new 4096-env SM baseline is a separate from-scratch run ending 2026-11-10 02:47:09 Shanghai; do not resume old stopped runs. Demo geometry is display-only; Sharpa reference previews are kinematic.

- Latest order (2026-11-09 late evening): reuse verified single-axis SM learning evidence; prioritize a bounded two-hour, matched PPO time-to-performance comparison on idle GPU3. Defer new multi-action and prior-control work. Single-task SM learning is verified; paper speedup and full pen adaptation are not. See [comparison protocol](docs/SM_PPO_VALIDATION_20261109.md).

## Work and evidence

- Use `source scripts/env.sh`; import Isaac Gym before Torch in simulator workers. Preserve action/controller order, terminal/reset boundaries and state/checkpoint provenance. Run checks relevant to the change; documentation edits do not require GPU training.
- Before training-code changes, inspect live processes and the run manifest; preserve active hashed sources. Keep original/best weights, cumulative evidence and unrelated edits. Never automatically resume user-stopped work or extend its resource budget. W&B and the old watchdog stay off unless requested.
- Latest explicit authorization (2026-11-09): a fresh 24-hour window, ending 2026-11-10 19:44:55 Asia/Shanghai, includes whole-duration random-phase fixes and acceptance, then automatic four-GPU SM/PPO promotion without another confirmation. Preserve mandatory checks and original deadline. See `work/authorizations/pen-comparison-20261109-v1.json` and [current evidence](docs/PEN_ADAPTATION_AND_QUEUE_20261109.md). Old windows and user-stopped runs remain closed.
- Separate planned, implemented, verified and measured results. Link evidence; old tests are not proof of the new loop. Update the responsible docs and contradictory entry-point summaries together; preserve historical results.
- Report concisely in Chinese: completed artifact, verification, next concrete task and any actual blocker. Proceed within existing authorization without adding routine confirmation steps.
