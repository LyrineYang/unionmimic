import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .common import digest, file_digest, save_checkpoint, write_json
from .models import load_world_model, CalibratedHeads, state_model_digest


def classification_metrics(prob, label, base_rate):
    prob, label = np.asarray(prob), np.asarray(label).astype(bool)
    positives = int(label.sum())
    order = np.argsort(-prob, kind='stable')
    precision = np.cumsum(label[order]) / np.arange(1, len(label)+1)
    ap = float((precision * label[order]).sum() / max(1, positives))
    threshold = float(np.quantile(prob[label], .2)) if positives else 1.0
    prediction = prob >= threshold
    return dict(positives=positives, prevalence=float(label.mean()),
                average_precision=ap, threshold=threshold,
                recall=float(prediction[label].mean()) if positives else 0.,
                false_positive_rate=float(prediction[~label].mean()) if (~label).any() else 0.,
                brier=float(np.mean((prob-label)**2)),
                baseline_brier=float(np.mean((base_rate-label)**2)))


@torch.no_grad()
def evaluate(heads, train, valid, device):
    pred_r, pred_p, pred_f = [], [], []
    for lo in range(0,len(valid['row']),256):
        feat = valid['features'][lo:lo+256].to(device)
        shape = feat.shape[:2]
        reward, prop, failure_logit = heads.predictions(feat.flatten(0,1))
        pred_r.append(reward.mode().reshape(shape).cpu())
        pred_p.append(prop.mode().reshape(*shape,73).cpu())
        pred_f.append(failure_logit.sigmoid().reshape(shape).cpu())
    rewards, props, failure = map(torch.cat,(pred_r,pred_p,pred_f))
    mean_r = float(train['reward'][train['mask']].float().mean())
    base_failure = float(train['failure'][train['mask']].float().mean())
    report = {}
    for h in (0, rewards.shape[1]-1):
        keep = valid['mask'][:,h]
        r, p, y = valid['reward'][keep,h], valid['prop'][keep,h], valid['failure'][keep,h]
        report[str(h+1)] = dict(
            reward_rmse=float((rewards[keep,h]-r).square().mean().sqrt()),
            reward_mean_baseline_rmse=float((mean_r-r).square().mean().sqrt()),
            reward_persistence_rmse=float((valid['start_reward'][keep]-r).square().mean().sqrt()),
            joint_rmse=float((props[keep,h,:22]-p[:,:22]).square().mean().sqrt()),
            joint_persistence_rmse=float((valid['start_prop'][keep,:22]-p[:,:22]).square().mean().sqrt()),
            contact_brier=float((props[keep,h,44:49].clamp(0,1)-p[:,44:49]).square().mean()),
            contact_persistence_brier=float((valid['start_prop'][keep,44:49]-p[:,44:49]).square().mean()),
            failure=classification_metrics(failure[keep,h].numpy(),y.numpy(),base_failure))
    checks = {}
    for horizon, m in report.items():
        checks['h'+horizon+'_reward_better_than_constant'] = m['reward_rmse'] < m['reward_mean_baseline_rmse']
        checks['h'+horizon+'_joint_better_than_persistence'] = m['joint_rmse'] < m['joint_persistence_rmse']
        fm = m['failure']
        checks['h'+horizon+'_failure_discriminates'] = (fm['positives'] >= 5 and
            fm['recall'] >= .7 and fm['false_positive_rate'] <= .15 and
            fm['average_precision'] > 1.5 * fm['prevalence'])
    policy_report = {}
    for h in (0,rewards.shape[1]-1):
        keep = valid['mask'][:,h] & ~valid['stress']
        if not bool(keep.any()):
            continue
        tr = train['mask'][:,h] & ~train['stress']
        mean = train['reward'][tr,h].float().mean()
        policy_report[str(h+1)] = dict(
            starts=int(keep.sum()),
            reward_rmse=float((rewards[keep,h]-valid['reward'][keep,h]).square().mean().sqrt()),
            reward_mean_baseline_rmse=float((mean-valid['reward'][keep,h]).square().mean().sqrt()),
            joint_rmse=float((props[keep,h,:22]-valid['prop'][keep,h,:22]).square().mean().sqrt()),
            joint_persistence_rmse=float((valid['start_prop'][keep,:22]-valid['prop'][keep,h,:22]).square().mean().sqrt()))
    return dict(horizons=report, policy_only=policy_report, checks=checks, passed=all(checks.values()))


def validate_checkpoint(args):
    cache = torch.load(str(args.cache),map_location='cpu')
    wm,saved,cfg = load_world_model(Path(cache['metadata']['dataset'])/'config.yaml',args.checkpoint,args.gpu)
    device = next(wm.parameters()).device
    assert state_model_digest(wm)==cache['metadata']['state_model_sha256']
    heads = CalibratedHeads(wm).to(device).eval()
    heads.failure.load_state_dict(saved['failure_head'])
    heads.failure_log_weight.fill_(saved['failure_log_weight'])
    report = evaluate(heads,cache['train'],cache['valid'],device)
    report.update(checkpoint=str(args.checkpoint),checkpoint_sha256=file_digest(args.checkpoint),
                  calibration_step=saved['calibration_step'],cache_metadata=cache['metadata'])
    write_json(args.output,report)
    print(json.dumps(report),flush=True)
    assert report['passed'], 'Prediction validation failed'


def calibrate(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    cache = torch.load(str(args.cache),map_location='cpu')
    metadata = cache['metadata']
    wm, saved, cfg = load_world_model(Path(metadata['dataset'])/'config.yaml', args.checkpoint, args.gpu)
    device = next(wm.parameters()).device
    assert state_model_digest(wm) == metadata['state_model_sha256'], 'Stale RSSM start cache'
    torch.manual_seed(args.seed)
    heads = CalibratedHeads(wm).to(device)
    if 'failure_head' in saved:
        heads.failure.load_state_dict(saved['failure_head'])
        heads.failure_log_weight.fill_(saved['failure_log_weight'])
    heads.freeze_body()
    params = [p for p in heads.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params,lr=args.lr)
    start_step = 0
    if args.resume:
        opt.load_state_dict(saved['calibration_optimizer'])
        start_step = int(saved['calibration_step'])
    train, valid = cache['train'], cache['valid']
    assert set(train['episode_id'].tolist()).isdisjoint(valid['episode_id'].tolist())
    assert int(train['failure'][train['mask']].sum()) > 10 and int(valid['failure'][valid['mask']].sum()) >= 20
    rate = float(train['failure'][train['mask']].float().mean())
    pos_weight = min(25., (1-rate)/max(rate,1e-6))
    if not args.resume:
        heads.failure_log_weight.fill_(math.log(pos_weight))
        if 'failure_head' not in saved:
            heads.failure[-1].bias.data.fill_(math.log(rate/(1-rate))+math.log(pos_weight))
    else:
        pos_weight = math.exp(float(heads.failure_log_weight))
    frozen_before = state_model_digest(wm)
    image_before = digest(wm.heads['decoder']._cnn)
    trainable_before = (digest(wm.heads['reward']),digest(wm.heads['decoder']._mlp),digest(heads.failure))
    before = evaluate(heads,train,valid,device)
    write_json(out/'before.json',before)
    # Cache features come solely from frozen-RSSM prior rollouts and executed
    # actions. Future observations are supervision, never model inputs.
    x = train['features'][train['mask']].to(device)
    r = train['reward'][train['mask']].to(device)
    p = train['prop'][train['mask']].to(device)
    fail = train['failure'][train['mask']].float().to(device)
    began = time.monotonic()
    if args.resume and 'calibration_rng' in saved:
        torch.set_rng_state(saved['calibration_rng']['cpu'])
        torch.cuda.set_rng_state(saved['calibration_rng']['cuda'],device)
    best_score = float('inf')
    best_report = None
    with open(out/'metrics.jsonl','a',buffering=1) as log:
        for update in range(start_step+1,start_step+args.updates+1):
            ids = torch.randint(len(x),(args.batch,),device=device)
            rd,pd,corrected = heads.predictions(x[ids])
            prediction = pd.mode()
            reward_loss = -rd.log_prob(r[ids]).mean()
            prop_loss = (10*(prediction[:,:22]-p[ids,:22]).square().mean() +
                         2*(prediction[:,44:49]-p[ids,44:49]).square().mean() +
                         .1*(prediction-p[ids]).square().mean())
            failure_loss = F.binary_cross_entropy_with_logits(
                corrected+heads.failure_log_weight,fail[ids],
                pos_weight=torch.tensor(pos_weight,device=device))
            loss = reward_loss + prop_loss + failure_loss
            assert bool(torch.isfinite(loss)), 'Nonfinite calibration loss'
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params,10.)
            assert bool(torch.isfinite(grad)) and float(grad)>0, 'Invalid calibration gradient'
            opt.step()
            if update % 50 == 0 or update==start_step+1:
                row = dict(phase='calibrate',update=update,loss=float(loss),
                           reward_loss=float(reward_loss),prop_loss=float(prop_loss),
                           failure_loss=float(failure_loss),grad_norm=float(grad),
                           seconds=round(time.monotonic()-began,1))
                log.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            if update % args.eval_every and update != start_step+args.updates:
                continue
            report = evaluate(heads,train,valid,device)
            report.update(update=update,train_examples=len(x),valid_starts=len(valid['row']),
                          state_model_sha256=frozen_before, cache_metadata=metadata,
                          encoder_rssm_unchanged=state_model_digest(wm)==frozen_before,
                          image_decoder_unchanged=digest(wm.heads['decoder']._cnn)==image_before,
                          reward_prop_failure_changed=all(a!=b for a,b in zip(
                              (digest(wm.heads['reward']),digest(wm.heads['decoder']._mlp),digest(heads.failure)),trainable_before)))
            assert report['encoder_rssm_unchanged'] and report['image_decoder_unchanged']
            score = sum(v['reward_rmse']/max(v['reward_mean_baseline_rmse'],1e-6) +
                        v['joint_rmse']/max(v['joint_persistence_rmse'],1e-6) +
                        v['failure']['false_positive_rate'] for v in report['horizons'].values())
            # Prefer any gate-passing checkpoint over a non-passing one.
            score += 100 * (not report['passed'])
            bundle = dict(model=saved['model'],world_model={'world_model':wm.state_dict()},
                          failure_head=heads.failure.state_dict(),failure_log_weight=float(heads.failure_log_weight),
                          calibration_step=update,calibration_optimizer=opt.state_dict(),
                          calibration_rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device)),
                          calibration_report=report,source_checkpoint_sha256=metadata['source_checkpoint_sha256'])
            for key in ('sm_state','sm_anchor','sm_report'):
                if key in saved:
                    bundle[key]=saved[key]
            save_checkpoint(out/'last.pt',bundle)
            write_json(out/'last.json',report)
            if score < best_score:
                best_score=score
                best_report=report
                save_checkpoint(out/'best.pt',bundle)
                write_json(out/'validation.json',report)
            print(json.dumps(dict(phase='validation',update=update,passed=report['passed'],
                                  checks=report['checks'],horizons=report['horizons'])),flush=True)
        write_json(out/'completed.json',dict(updates=start_step+args.updates,
                   seconds=time.monotonic()-began,best_passed=best_report['passed']))
