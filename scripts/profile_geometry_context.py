"""Isolated encoder/readout throughput; excludes cameras and simulator cost."""
import unionmimic.runtime
import argparse
import time
from pathlib import Path

import h5py
import torch
from unionmimic.models import load_world_model
from unionmimic.geometry_context import PositionReadout
from unionmimic.common import write_json, digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--gpu', type=int, default=1); p.add_argument('--out', required=True)
    a = p.parse_args(); out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.cuda.set_device(a.gpu); dev = torch.device('cuda', a.gpu)
    source = ROOT/'work/analysis/geometry-heldout-20261109-v1'
    wm, saved, _ = load_world_model(source/'config.yaml', gpu=a.gpu)
    encoder = wm.encoder.eval().requires_grad_(False); del wm, saved
    head_ck = torch.load(ROOT/'work/analysis/geometry-readout-20261109-v1/pretrained.pt', map_location='cpu')
    assert digest(encoder) == head_ck['encoder_sha256']
    head = PositionReadout.from_checkpoint(head_ck).to(dev)
    with h5py.File(source/'episodes.h5') as f:
        ep = f[next(iter(f))]
        obs = dict(prop=torch.as_tensor(ep['prop'][:256]).to(dev), image=torch.as_tensor(ep['depth'][:256]).to(dev))
    assert len(obs['prop']) == 256
    history = obs['prop'][:, None].expand(-1, 10, -1).contiguous()
    rows = []
    with torch.no_grad():
        for batch in (16, 256, 4096):
            chunk = min(batch, 256)
            x = {k:v[:chunk] for k,v in obs.items()}; hist = history[:chunk]
            def step():
                for _ in range(batch//chunk):
                    head(hist, encoder(x))
            for _ in range(5): step()
            torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            for _ in range(20): step()
            torch.cuda.synchronize(); seconds = (time.perf_counter()-t)/20
            rows.append(dict(batch=batch, chunk=chunk, seconds=seconds, rows_per_s=batch/seconds,
                             peak_increment_mib=(torch.cuda.max_memory_allocated()-base)/2**20))
    write_json(out/'result.json', dict(rows=rows, encoder_parameters=sum(p.numel() for p in encoder.parameters()),
        readout_coefficients=head.weight.numel()+head.offset.numel(), device=torch.cuda.get_device_name(),
        scope='FP32 frozen encoder + readout, device-resident inputs, no backward, no simulator/camera/transfer; 4096 is 16 chunks of 256.',
        end_to_end_training_speed_measured=False))


if __name__ == '__main__': main()
