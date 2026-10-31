# Research insights

**Can a world model built for sensing also provide useful gradients for control?**

UnionMimic starts from an existing visuotactile RSSM and policy. Its immediate goal is a working short-horizon policy-training loop and a measured comparison with PPO. The ideas below guide the implementation; they are hypotheses, not reported findings.

## One model, two operations

`observe` uses an action and new sensor evidence to update a belief about the present. `imagine` uses an action and the current belief to predict a future without new evidence. The RSSM's posterior and prior provide a practical boundary for this reuse.

The adaptation does not require translating two independently trained latent spaces. The actor still produces actions; the shared RSSM supplies state estimates and differentiable predictions.

**Build:** explicit observation and imagination paths restored from the same checkpoint. **Validate:** four-step prior predictions, followed by actor updates evaluated in the original simulator.

## Reconstruction is not a control objective

A state estimator receives corrective observations at every step. An imagined rollout does not. Low reconstruction error can therefore coexist with poor predictions of reward, contact changes, and drops.

Prediction accuracy also does not establish gradient quality. Near a contact switch, a smooth learned approximation may point toward an action that improves predicted return but hurts physical return.

**Measure:** one- and four-step prior errors, reward calibration, failure recall and false-positive rate, and the gap between imagined and simulator returns. A later diagnostic can compare predicted improvement directions with paired simulator action perturbations, using repeated trials and common random seeds. Do not assume contact dynamics are smooth.

**If it fails:** recalibrate heads, refresh on-policy data, or shorten the horizon before spending more time on actor optimization.

## Interfaces are part of the method

The actor's command is not the action received by the RSSM. Noise, clipping, smoothing, and accumulated joint targets sit between them. The actor also uses a four-frame observation history alongside the model feature.

Ignoring these details trains against the wrong control system. Recording a reset observation as the outcome of a terminal action creates an equally misleading model.

**Build:** episode-aware transition records, full recurrent-state restoration, a task-matched action adapter, and differentiable actor inference that preserves the released normalization statistics. These are concrete engineering contributions, not a claim of a new algorithm.

## Short imagination is a starting constraint

Start with H = 4, approximately 0.4 seconds at the task's roughly 10 Hz control rate. Physical duration matters more than copying a step count from another robot.

Compare shorter diagnostic horizons first; test H = 8 only when prior predictions are usable. Adaptive truncation is a later experiment, triggered by measured error growth near contact changes. Predictive variance alone is not calibrated uncertainty.

## Count the whole loop

Saving simulator interactions may still increase total cost if model fitting and imagination are expensive. Count collection, calibration, model updates, actor/critic updates, and evaluation separately, and report their total.

Compare PPO and SM fine-tuning from the same released checkpoint, on the same object and perturbation distribution, with paired seeds. Frozen replay is a capability reference, not a training-efficiency baseline.

## Directions after the first loop

| Direction | Trigger | Smallest useful experiment |
| --- | --- | --- |
| Cross-object reuse | The single-object loop works | Frozen model vs fine-tuning vs training from scratch on a second object; count all adaptation cost |
| Richer touch | Binary contact misses a recurring failure | Add a supported continuous signal with matched data; verify simulator fidelity |
| Adaptive horizon | Prediction error varies across contact regimes | Fixed horizon vs truncation calibrated on held-out episodes |
| Shared body/hand/leg dynamics | Multiple working tasks and embodiments exist | Define physical targets and embodiment-specific interfaces before sharing latent dynamics |

Matching token counts or latent dimensions does not make dynamics interchangeable. Cross-embodiment experiments and real-robot deployment are outside the first milestone.

## Source-plan corrections

A detached policy feature does not imply a permanently frozen world model: WM-Craftnet's training procedure includes model updates from replay. [WM-Craftnet training details](https://arxiv.org/html/2609.07002v1)

SuperMimic motivates policy gradients through short learned-model rollouts. Porting the mechanism requires a rollout objective, reward and failure handling, and value learning; removing `detach` alone is insufficient. [SuperMimic](https://arxiv.org/abs/2610.28480)

The design above adapts those ideas to this hand and sensor stack. Upstream humanoid speedups are not forecasts for UnionMimic.
