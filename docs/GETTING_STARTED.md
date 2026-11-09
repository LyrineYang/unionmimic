# Getting started

Current baseline development uses the pinned LEAP native recipe, followed by Sharpa adaptation. Use [the native execution record](LEAP_NATIVE_EXECUTION_20261103.md) and [its isolated environment](ENVIRONMENT.md) for that path. The Isaac Gym steps below preserve the original Sharpa runtime check and historical experiments.

## Runtime prerequisites

Verified stack: Linux, Tesla V100-SXM2-32GB, Python 3.8.20, PyTorch 2.1.0 / CUDA 11.8, Isaac Gym Preview 4, PyTorch3D 0.7.5, driver 530.30.02. CUDA alone is insufficient: depth cameras need a working Vulkan/EGL path.

The bootstrap expects GCC 11 and `/usr/local/cuda-11.8`, and builds PyTorch3D for compute capability 7.0. Review these settings for another GPU. Obtain Isaac Gym through NVIDIA's distribution and applicable terms. Download availability can vary; cached artifacts are accepted only when checksums match.

## Reproduce the check

From the repository root:

```bash
bash scripts/bootstrap.sh
bash scripts/run_step1.sh --gpu 0 --steps 200
```

Choose an available GPU. The launcher uses explicit physical indices to align CUDA and Vulkan; do not separately remap `CUDA_VISIBLE_DEVICES` without auditing the mapping.

Use `--stage cuda`, `--stage gym`, or `--stage policy` for individual checks. Outputs go to `work/results/step1/`. Inspect actual sensor and policy evidence, not just successful imports or exit codes.

## Local resources

`source scripts/env.sh` sets environment and cache paths. `.venv/` and `work/` contain dependencies, upstream code, weights, data and logs; they are excluded from Git. Proxy settings belong in ignored `config/network.local.env`; see [the example](../config/network.example.env). Never commit credentials.

For the measured package lock, environment verification, candidate-environment rebuilds and W&B configuration, see [environment management](ENVIRONMENT.md). Existing locked environments are checked rather than reinstalled by bootstrap.

Isaac Gym's convex-decomposition cache uses its default user directory. The launcher retains the NVIDIA EGL library mapping to address the observed rendering-thread exit issue while still cleaning up the simulator.

## Training development

The stable check above replays the original policy without parameter updates. Training integration is separate work. Read [architecture](ARCHITECTURE.md), [experiments](EXPERIMENTS.md), and [status](STATUS.md) before treating a training entry point as validated.

Do not replace the simulator or remove required depth/contact inputs to make a partial check appear complete. Record unavailable components precisely.

## Website preview

```bash
cd website
python3 -m http.server 8080 --bind 127.0.0.1
```

Open `http://localhost:8080` on the same machine. [WEBSITE.md](WEBSITE.md) covers remote forwarding, packaging and future publication.
