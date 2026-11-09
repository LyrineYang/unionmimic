"""Bounded checks for the pinned native runtime; no policy training."""
import argparse
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import torch
import warp as wp


@wp.kernel
def double_values(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = 2.0 * x[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert torch.__version__.split('+')[0] == '2.11.0', torch.__version__
    assert torch.version.cuda == '12.6', torch.version.cuda
    assert 'sm_70' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
    device = torch.device('cuda:0')
    torch.manual_seed(0)
    x = torch.randn(256, 256, device=device, requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
    wp.init()
    a = wp.array(np.arange(256, dtype=np.float32), device='cuda:0')
    b = wp.zeros(256, dtype=float, device='cuda:0')
    wp.launch(double_values, dim=256, inputs=[a, b], device='cuda:0')
    wp.synchronize()
    np.testing.assert_array_equal(b.numpy(), 2 * np.arange(256, dtype=np.float32))
    libc = ctypes.CDLL(None)
    libc.gnu_get_libc_version.restype = ctypes.c_char_p
    result = dict(
        passed=True, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
        compiled_arches=torch.cuda.get_arch_list(), libc=libc.gnu_get_libc_version().decode(),
        torch_backward_finite=True, warp_kernel_exact=True,
        packages={name: importlib.metadata.version(name) for name in
                  ['torch', 'numpy', 'mjlab', 'mujoco', 'mujoco-warp', 'warp-lang', 'rsl-rl-lib']},
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
