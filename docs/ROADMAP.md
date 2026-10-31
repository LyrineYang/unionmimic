# Roadmap

**Deliver the full WM + SM adaptation, then compare it with PPO.**

| Stage | Work | Deliverable for the next stage | Evidence state |
| --- | --- | --- | --- |
| 01 · Restore | Verify released weights, physics, sensors, RSSM and replay | Reproducible original-policy runtime | Recorded acceptance passed |
| 02 · Predict | Collect aligned episodes, restore starts, calibrate four-step prediction | Validated data and model checkpoint | Development in progress; acceptance pending |
| 03 · Learn | Match actor/actions; add imagination, backpropagation and value learning | Updated policy, save/resume, physical evaluation | Integration in progress; acceptance pending |
| 04 · Compare | Fine-tune PPO and SM from the same weights | Paired quality/cost curves and failure analysis | Results pending |
| 05 · Present | Package commands, evidence and real evaluation media | Reproducible release and expanded website | Documentation and website foundation available |

This is documented acceptance, not a live job monitor. A running process or source file does not close a stage. Update [STATUS.md](STATUS.md) with evidence when a stage passes.

The first follow-up is a second object on the same hand. Longer-term body/hand/leg sharing requires additional working tasks and embodiment interfaces. Cross-embodiment and real-robot capability are not claimed here.
