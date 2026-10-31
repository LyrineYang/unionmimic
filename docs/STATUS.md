# Evidence and status

**Runtime verified; comparative training evidence pending.**

Documentation snapshot: **1 November 2026 (Asia/Shanghai)**. The runtime check completed at `2026-10-31T17:38:52Z`, or 1 November, 01:38 in Shanghai. Training development may advance separately; only checked-in acceptance evidence is represented here.

| Check | Recorded result |
| --- | --- |
| Hardware | One Tesla V100-SXM2-32GB |
| CUDA | FP32 forward/backward; finite, nonzero gradient |
| Physics and camera | 100 physics steps; GPU depth tensor, 72 × 96; 6,048 valid pixels |
| Checkpoint | Released z-axis policy and world model loaded strictly |
| Replay | One environment, `set_obj6_block_corner`, seed 42; 200 control steps |
| Replay time | 36.1288 seconds, including checks; excluding first load and asset preparation |
| Reward | 446.4927 over the short replay; not a comparative benchmark |
| Touch | Five fingertip contact channels; 232 contact-state changes |
| Reset and isolation | Explicit reset passed; policy/model hashes unchanged |
| Exit | CUDA, physics and policy processes each returned 0 |

Source: [step1-summary.json](evidence/step1-summary.json). Details: [dependencies](evidence/environment-freeze.txt), [configuration](evidence/policy-config.yaml), [original acceptance report](STEP1.md) (Chinese, host-date convention).

The PyTorch allocator reported 171.83 MiB peak, excluding PhysX/Vulkan allocations. It is not total VRAM consumption. This run did not complete a full 500-step episode or measure success rate.

## Sensor evidence

| First recorded depth frame | Last recorded depth frame |
| --- | --- |
| ![First depth frame](evidence/depth-0000.png) | ![Last depth frame](evidence/depth-0199.png) |

These are actual depth samples, not RGB robot renders or trained-policy demonstrations.

## Evidence still needed

- Accepted aligned data and four-step reward/failure prediction.
- Validated SM-updated policy and save/resume behavior.
- Paired PPO/SM learning curves, quality thresholds and cost measurements.
- Cross-object, cross-embodiment or real-robot transfer.

Promote a proposed capability to a result only when its evaluation artifacts are available.
