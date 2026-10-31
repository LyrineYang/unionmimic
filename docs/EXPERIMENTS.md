# Experiment design

**Measure whether SM fine-tuning improves task quality or total adaptation cost under matched conditions.** Comparative results are pending.

## First comparison

Fix the hand, single object, released checkpoint, z-axis objective, sensor inputs, reward, action mapping, and perturbation distribution. If the original task is saturated, choose a harder setting before comparing methods.

| Run | Role |
| --- | --- |
| Released policy, frozen | Starting capability reference |
| PPO fine-tuning | Original optimization route and training-cost baseline |
| RSSM + SM fine-tuning | Short imagined-rollout gradients under test |

Keep trainable modules and model-update budgets explicit. Count SM's extra calibration. Report total incremental cost and any shared preparation cost. Frozen inference is not a training-speed baseline.

Debug with one seed, then use two paired training seeds and 100–200 held-out evaluation episodes per trained policy. Match evaluation seeds and resets. Two seeds support an exploratory comparison, not a strong significance claim. Show per-seed results; do not treat evaluation episodes as independent training seeds.

## Metrics

| Question | Record |
| --- | --- |
| Does control improve? | Rotation progress, drop rate, episode length and original task return |
| Is adaptation cheaper? | New transitions, wall-clock time and GPU-hours to a predefined quality threshold |
| What happens at fixed budget? | Quality at matched compute and separately at matched interactions |
| Where does time go? | Collection, calibration, model fitting, actor/critic updates, evaluation, startup |
| Is prediction usable? | One-/four-step prior error, reward error, failure recall and false-positive rate |
| Is the actor exploiting model error? | Imagined-return improvement versus simulator-return change |

Select numeric quality thresholds before comparing methods; the runtime check does not define them. Keep raw and scaled rewards separate. Record GPU, environment count, graphics mapping, precision and warm-up policy. Do not extrapolate GPU times using peak FLOPS.

## Before the first actor update

1. Check time alignment, terminal observations and episode-level splits.
2. Restore RSSM and controller states from real history.
3. Compare reward/failure predictions against constant or persistence references on held-out episodes, including failures.
4. Match adapted deterministic actions to the original player on real observations.
5. Check finite actor gradients and changed actor parameters; model and critic stay fixed during the actor step.
6. Save, reload and evaluate in the original simulator.

These diagnose implementation errors; passing them does not demonstrate an efficiency gain.

## Follow-up experiments

| Hypothesis | Change one factor | Decision evidence |
| --- | --- | --- |
| A short prior can train the actor | PPO vs SM with matched initialization | Physical quality and total cost |
| Longer imagination helps while predictions remain useful | H = 4, then H = 8 | Contact-conditioned error and simulator performance |
| Touch improves prediction around contact changes | Matched model with/without contact input | Prediction error and physical drop/recovery behavior |
| Dynamics reuse saves adaptation cost | Second object: frozen, fine-tuned, from scratch | Additional data and compute at matched quality |

Run these after the integrated comparison. Continuous tactile fields, other embodiments and hardware deployment remain later work.

## Report all outcomes

Keep failed runs, divergence and threshold misses. Separate alignment errors, model errors, gradient instability and runtime cost. A working integration and an unsuccessful efficiency hypothesis are distinct outcomes.

Each comparison should link configuration, code revision, checkpoint hash, seeds, per-seed metrics, cost breakdown and failure examples. Videos need playback speed, policy identity, task conditions and failure frequency. Until then, the homepage shows runtime evidence and this protocol only.
