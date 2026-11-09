import math
import torch
import torch.nn.functional as F
H_OBS=10
def set_grad(mods,flag):
  for m in mods:
    for p in m.parameters():p.requires_grad_(flag)

def fit_update(wm,err,replay,opt_wm,args,dev,N,norm_c):
  st = {}
  # 1. world model
  for _ in range(args.wm_steps):
    g, e = replay.sample_windows(args.wm_batch, args.wm_k)
    s0, acts, snx, cx, dn = replay.window(g, e, args.wm_k)
    c = norm_c(cx)
    sh, ld = s0, 0.0
    for j in range(args.wm_k):
      sh, _ = wm(sh, acts[j], c, scale=replay.plant[(g+j)%replay.L,e], external_force=snx[j,:,145:148])
      ld = ld + err(sh, snx[j]).mean()
    ld = ld / args.wm_k
    with torch.no_grad():
      from in_hand_rotation_mjlab.world_model.model import geodesic, quat_conj, quat_mul, twist_z
      Sx = wm.S
      q0c = quat_conj(s0[..., Sx["quat"]])
      tw = twist_z(quat_mul(sh[..., Sx["quat"]], q0c)) - twist_z(quat_mul(snx[-1][..., Sx["quat"]], q0c))
      tw = torch.remainder(tw + math.pi, 2 * math.pi) - math.pi
      acc_tw = float(tw.abs().mean() * 180 / math.pi)
      acc_pos = float((sh[..., Sx["p"]] - snx[-1][..., Sx["p"]]).norm(dim=-1).mean() * 1e3)
    # one-step: termination head (terminated rows oversampled to a quarter) + torque / work head
    rr, ee = replay.failure_rows(args.wm_batch)
    s1, a1, n1, c1 = replay.s[rr, ee], replay.a[rr, ee], replay.s_next[rr, ee], norm_c(replay.ctx[rr, ee])
    cmd1 = a1
    with torch.no_grad():
      p1, _ = wm(s1, cmd1, c1, scale=replay.plant[rr,ee], external_force=n1[:,145:148])
    lt = F.binary_cross_entropy_with_logits(wm.failure_logit(p1, c1),
                                            (replay.done[rr, ee] == 1).float())
    la = wm.sensor_loss(s1, n1, c1)
    opt_wm.zero_grad(set_to_none=True)
    (ld + lt + la).backward()
    torch.nn.utils.clip_grad_norm_(list(wm.parameters()), 1.0)
    assert torch.isfinite(ld+lt+la), "Nonfinite model objective"
    assert all(torch.isfinite(p.grad).all() for p in wm.parameters() if p.grad is not None), "Nonfinite model gradient"
    opt_wm.step()
  st.update(wm=float(ld), term=float(lt), aux=float(la), wm_twist_deg=acc_tw, wm_pos_mm=acc_pos)
  return st

def actor_critic_update(actor,critic,wm,task,replay,opt_pi,opt_v,args,n_up,dev,norm_c,dt,r_max):
  st = {}
  # 2. actor through imagination
  fin = args.imag_noise if args.imag_noise_final is None else args.imag_noise_final
  frac = min(1.0, max(0, n_up - args.wm_only) / max(1, args.imag_noise_updates))
  imag_sigma = args.imag_noise + (fin - args.imag_noise) * frac
  set_grad([wm], False)
  g, e = replay.sample_starts(args.starts, H_OBS)
  r0 = g % replay.L
  s, info, ctx, plant = replay.s[r0, e], replay.info[r0, e], norm_c(replay.ctx[r0, e]), replay.plant[r0, e]
  frames = replay.history("frame", g, e, H_OBS)
  assert len(g)==args.starts, "Insufficient full-history actor starts"
  dprev = torch.zeros(len(g),3,device=dev)
  J, surv, disc = 0.0, torch.ones(len(g), device=dev), 1.0
  feats, rews, conts = [], [], []
  for h in range(args.horizon):
    feats.append(task.critic_features(s, info, dprev, ctx).detach())
    a = actor(task.flatten(frames), noise=imag_sigma)
    command = task.imagined_command(a, ctx)
    s2, logit = wm(s, command, ctx)
    terms, dprev = task.reward_rates(s, s2, info, dprev, ctx)
    r = sum(terms.values()) * dt
    p = torch.sigmoid(logit)
    J = J + disc * surv * r
    rews.append(r.detach()); conts.append((1 - p).detach())
    surv = surv * (1 - p)
    disc *= args.gamma
    frame, s2 = task.imagined_frame(s2, ctx)
    frames = torch.cat([frames[:, 1:], frame[:, None]], 1)
    s = s2
  fH = task.critic_features(s, info, dprev, ctx)
  J = J + disc * surv * critic(fH)
  loss_pi = -J.mean()
  opt_pi.zero_grad(set_to_none=True)
  loss_pi.backward()
  gn = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
  torch.nn.utils.clip_grad_value_(actor.parameters(), 0.1)
  ok = all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
  assert ok and torch.isfinite(loss_pi) and torch.isfinite(gn), "Nonfinite actor update"
  opt_pi.step()
  set_grad([wm], True)
  # 3. critic: TD(lambda) on the detached rollout
  with torch.no_grad():
    feats.append(fH.detach())
    X = torch.stack(feats)  # (H+1, B, d)
    critic.norm.update(X)
    V = critic(X)
    Rt = V[-1]
    tgt = []
    for h in reversed(range(args.horizon)):
      Rt = rews[h] + args.gamma * conts[h] * ((1 - args.lam) * V[h + 1] + args.lam * Rt)
      tgt.append(Rt)
    T = torch.stack(tgt[::-1]).clamp(-r_max / (1 - args.gamma), r_max / (1 - args.gamma))
  Xf, Tf = X[:-1].reshape(-1, X.shape[-1]), T.reshape(-1)
  lv = 0.0
  for _ in range(5):
    perm = torch.randperm(len(Xf), device=dev)
    for mb in perm.chunk(4):
      l = (critic(Xf[mb]) - Tf[mb]).pow(2).mean()
      opt_v.zero_grad(set_to_none=True)
      l.backward()
      torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
      assert torch.isfinite(l) and all(torch.isfinite(p.grad).all() for p in critic.parameters() if p.grad is not None), "Nonfinite critic update"
      opt_v.step()
      lv = float(l)
  st.update(J=float(J.mean()), imag_r=float(torch.stack(rews).mean() / dt), v=lv, gn=float(gn),
            surv=float(surv.mean()), finite=ok)
  return st
