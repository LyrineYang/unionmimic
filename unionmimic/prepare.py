import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from .common import save_checkpoint, write_json, file_digest
from .models import load_world_model, state_model_digest


def verify_state_provenance(recorded_hash, current_hash, recompute=False):
    if str(recorded_hash).split(':')[0] != current_hash.split(':')[0]:
        raise ValueError('Encoder changed: re-encode raw observations before preparing this dataset')
    if recorded_hash != current_hash and not recompute:
        raise ValueError('RSSM changed: explicitly rebuild posterior states with --recompute-state')


def future_indices(length, warmup, horizon, device):
    starts = torch.arange(warmup, length-1, device=device)
    future = starts[:,None] + torch.arange(1,horizon+1,device=device)[None]
    return starts, future.clamp(max=length-1), future < length


@torch.no_grad()
def recover_episode(wm, group, device, force_recorded_samples=True, batch_size=16):
    embed = torch.as_tensor(group['embed'][:], device=device)
    action = torch.as_tensor(group['executed_action'][:], device=device)
    codes = F.one_hot(torch.as_tensor(group['post_index'][:].astype('int64'), device=device),
                      wm.dynamics._discrete).float()
    # Preserve the recording GEMM batch shape and lane. Batch-1 GEMV roundoff
    # otherwise accumulates across hundreds of recurrent transitions.
    lane = int(group.attrs['env_id'])
    first = torch.zeros(batch_size, device=device)
    state = None
    states = []
    for t in range(len(embed)):
        first.fill_(float(t == 0))
        state, _ = wm.dynamics.obs_step(state, action[t:t+1].repeat(batch_size,1),
                                       embed[t:t+1].repeat(batch_size,1), first,
                                       sample=not force_recorded_samples)
        if force_recorded_samples:
            # Replaying the recorded categorical draws reproduces the original
            # stochastic inference trajectory, including all history before t.
            state['stoch'] = codes[t:t+1].repeat(batch_size,1,1)
        states.append({k:v[lane:lane+1].clone() for k,v in state.items()})
    return {k:torch.cat([s[k] for s in states]) for k in state}


@torch.no_grad()
def prepare(args):
    folder = Path(args.data).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    wm, saved, cfg = load_world_model(folder/'config.yaml', args.checkpoint, args.gpu)
    device = next(wm.parameters()).device
    torch.manual_seed(args.seed)
    state_hash = state_model_digest(wm)
    recompute = getattr(args, 'recompute_state', False)
    pools = {'train': [], 'valid': []}
    max_deter = max_logit = max_embed = 0.0
    episode_ids = {'train': [], 'valid': []}
    start = time.monotonic()
    batch_size=int(cfg['task']['env']['numEnvs'])
    with h5py.File(folder/'episodes.h5', 'r') as f:
        assert f.attrs['complete'], 'Collection is still in progress'
        verify_state_provenance(f.attrs['state_model_sha256'], state_hash, recompute)
        for ordinal, key in enumerate(f):
            group = f[key]
            n = len(group['reward'])
            if n <= args.warmup + 1:
                continue
            state = recover_episode(wm, group, device, force_recorded_samples=not recompute,batch_size=batch_size)
            if not recompute:
                max_deter = max(max_deter, float((state['deter']-torch.tensor(group['post_deter'][:],device=device)).abs().max()))
                max_logit = max(max_logit, float((state['logit']-torch.tensor(group['post_logit'][:],device=device)).abs().max()))
            if ordinal < 3:
                ix = np.arange(min(n, 12))
                emb = wm.encoder({'prop':torch.tensor(group['prop'][ix],device=device),
                                  'image':torch.tensor(group['depth'][ix],device=device)})
                max_embed = max(max_embed, float((emb-torch.tensor(group['embed'][ix],device=device)).abs().max()))
                torch.manual_seed(123)
                a = recover_episode(wm, group, device, False,batch_size)
                torch.manual_seed(123)
                b = recover_episode(wm, group, device, False,batch_size)
                assert all(torch.equal(a[k], b[k]) for k in a), 'Seeded history replay is not reproducible'
            starts, ids, mask = future_indices(n,args.warmup,args.horizon,device)
            block = { 'state_'+k:v[starts].cpu() for k,v in state.items() }
            # Retain starts near termination. Mask padding instead of dropping
            # all H1 failures by requiring four future nonterminal rows.
            block['mask'] = mask.cpu()
            for field in ('prop','reward','failure','timeout','executed_action'):
                block[field] = torch.as_tensor(group[field][:],device=device)[ids].cpu()
            for field in ('prop','reward','history','policy_obs','targets','last_actions'):
                block['start_'+field] = torch.as_tensor(group[field][:],device=device)[starts].cpu()
            block['episode_id'] = torch.full((len(starts),),int(key),dtype=torch.long)
            block['row'] = starts.cpu()
            block['stress'] = torch.full((len(starts),),group.attrs['behavior']!='policy',dtype=torch.bool)
            split = group.attrs['split']
            pools[split].append(block)
            episode_ids[split].append(key)
            if ordinal % 10 == 0:
                print(json.dumps(dict(phase='recover', episodes=ordinal+1, max_deter_error=max_deter)),flush=True)
        lower, upper = np.asarray(f.attrs['lower_limits']).tolist(), np.asarray(f.attrs['upper_limits']).tolist()
    assert max_deter < 2e-4 and max_logit < 2e-4 and max_embed < 2e-4, (max_deter,max_logit,max_embed)
    assert not set(episode_ids['train']) & set(episode_ids['valid'])
    caches = {}
    for split, blocks in pools.items():
        assert blocks, 'No complete episodes in ' + split
        cache = {k:torch.cat([b[k] for b in blocks]) for k in blocks[0]}
        # Keep the full holdout, including rare terminal windows. Randomly
        # capping validation can remove its failure examples as data grows.
        limit = args.max_starts if split=='train' else len(cache['row'])
        if len(cache['row']) > limit:
            torch.manual_seed(args.seed)
            selected = torch.randperm(len(cache['row']))[:limit]
            cache = {k:v[selected] for k,v in cache.items()}
        features = []
        for lo in range(0,len(cache['row']),256):
            hi = lo+256
            state = {k:cache['state_'+k][lo:hi].to(device) for k in ('deter','stoch','logit')}
            seq = []
            for h in range(args.horizon):
                state = wm.dynamics.img_step(state, cache['executed_action'][lo:hi,h].to(device))
                seq.append(wm.dynamics.get_feat(state).cpu())
            features.append(torch.stack(seq,dim=1))
        cache['features'] = torch.cat(features)
        caches[split] = cache
    metadata = dict(state_model_sha256=state_hash, source_checkpoint_sha256=file_digest(args.checkpoint),
                    dataset=str(folder), dataset_sha256=file_digest(folder/'episodes.h5'),
                    horizon=args.horizon, warmup_minimum=args.warmup,
                    full_episode_history_used=True, seeded_replay_exact=True,
                    posterior_mode='recomputed_from_full_history' if recompute else 'replayed_recorded_categorical_samples',
                    recorded_posterior_reproduced=not recompute,
                    max_deter_error=None if recompute else max_deter,
                    max_logit_error=None if recompute else max_logit,max_embed_error=max_embed,
                    episodes=episode_ids, lower_limits=lower, upper_limits=upper,
                    seconds=time.monotonic()-start)
    # Only observed history initializes the state. The future rollout uses
    # recorded actions alone: no future image, prop, embedding or posterior.
    save_checkpoint(output/'cache.pt',dict(metadata=metadata,**caches))
    report = {**metadata,'starts':{s:len(caches[s]['row']) for s in caches},
              'future_failure_labels':{s:int((caches[s]['failure'] & caches[s]['mask']).sum()) for s in caches}}
    write_json(output/'recovery.json',report)
    print(json.dumps(report),flush=True)
