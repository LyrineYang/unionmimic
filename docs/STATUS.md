# Evidence and status

**11月9日晚复现结论对齐：** SM完整在线闭环和Sharpa角块单轴旋转学习已经验证成功；尚未完成的是公平PPO训练速度比较、完整多动作与先验控制收益。用户要求先做快速SM/PPO成本对照，暂缓后两项。已安排GPU3两小时PPO，先通过最新源码操作检查；GPU0现有SM预算不变。[已有结果与比较协议](SM_PPO_VALIDATION_20261109.md)。

**11月9日晚最新推进：** 已完成冻结编码器的小型位置读出、三组同模态随机编码器对照与真实／想象误差检查，并交付实时几何Demo及52秒中文成片。独立测试位置误差4.00→2.18 mm，尚无控制收益。另开新的4096环境SM随机起点基线，最长至11月10日02:47:09上海；350轮短试跑通过后自动继续，不恢复旧停止任务。50Hz转笔物理门槛仍未通过。详见[几何先验与后备方案](GEOMETRY_AND_DEMO_BACKUP_20261109.md)。

**11月9日三阶段先验机制实验：** 用户授权直接完成。相容10Hz角块上检验附加信息、随机策略三组控制、负结果诊断；原50Hz转笔物理门槛保持不变，不以此替代完整适配。三个阶段已完成：三组400轮短训均未学会持续旋转；离线物体位置估计发现可用几何信息，尚无控制收益。详见 [执行与结果](CONTEXT_STUDY_20261109.md)。

**11月9日先验短测：** 已完成冻结RSSM成本、触觉信息探针与上下文递推验证。4096×16分支约0.190s/额外3.13GiB；信息探针未超过10帧历史。新笔候选首帧接触仍未过验收，长训未启动。[实测与边界](FROZEN_CONTEXT_AUDIT_20261109.md)。

**2026-11-09 重新授权：** 本次24小时窗口固定至11月10日19:44:55（上海），包括修复、验收与训练；全部验收通过后自动启动四卡SM/PPO，无需再次确认。新候选整段相位首步通过354/416（85.1%），尚未达到90%，正修复接触与重定向；长训尚未启动。[现场与计划](PEN_ADAPTATION_AND_QUEUE_20261109.md) · [可展示demo](DEMOS.md)。


**11月4日最新决定：完整Sharpa多动作适配优先。** 角块SM/PPO长训已于11:28:50停止，权重和历史评估保留，不自动恢复。50Hz、四任务物理接口与模拟位姿观测已有短测；单笔抓取未过验收，完整参考命令、学习器/想象接口仍待完成。先完成这些适配，再比较SM/PPO，之后接WM-Craftnet。[验收缺口与执行顺序](SHARPA_FULL_ADAPTATION_20261104.md)。以下日期记录属于历史快照。


**23:53已启动：** GPU2／tmux `unionmimic-sharpa-leap`，运行目录 `work/runs/sharpa-leap-12h-20261103-v1`。这是LEAP学习器的Sharpa22适配，非纯LEAP任务；最长至11月4日11:53:34（上海），包含预热与评估。用户已要求提前结束完整试跑，保留300轮算法预热及最小更新检查。v3完整操作检查与独立512首回合评估通过；只认证执行/一致性，单次更新策略失败率100%，未验证学习收益。[启动证据](evidence/sharpa-leap-launch-20261103.json)。


**23:52用户追加指令：** 终止完整短试跑并启动Sharpa长训。保留已完成操作检查和独立部署评估；执行提前晋级，保留300轮算法预热与最小更新/回采检查，取消完整350轮短试跑等待。此为明确授权，不再等待新的确认；最长12小时不变。LEAP原生试跑最终评估重试已停止，原531轮训练和所有权重保留，不自动恢复。


**11月3日23:51最新执行规则：** 用户已明确授权“适配核验后启动训练”。本轮固定LEAP配方的Sharpa任务可在操作检查、全规模短试跑及独立评估通过后，同进程继续至最多12小时；此前“尚未授权”仅为历史状态。提前缩短试跑尚未执行。Wuji参考动作与WSM尚未接入。见[当前完整方案与证据](SHARPA_LEAP_TRAINING_20261103.md)。


**11月3日23:17 Sharpa适配接口：** 4096环境×80控制步检查已通过，327096条有效转移，动作/目标误差0；原版LEAP F以3×1024、batch2048、k8和replay1000完成10次物理损失更新，输入梯度有效。仅完成物理状态、控制器和F接口，尚无完整Sharpa在线SM/actor/critic验收，不具备长训放行条件。LEAP发布权重首次评估超时，未算通过；正在相同协议重试一次。详见[实测与剩余适配](LEAP_NATIVE_EXECUTION_20261103.md#2317适配接口实测)。

**11月3日22:53原生试跑：** 固定LEAP提交`0ab03eb`的完整4096环境配方已在GPU1/tmux启动45分钟短训；actor/critic/F从头初始化。33项包测试、基础CUDA/Warp检查通过，原生奖励核对正常退出但有已记录近似差异；发布权重评测在GPU2执行。学习与Sharpa适配尚未验收，新长训未获确认。预计23:40—23:50验收，见[运行目录、差异及监督命令](LEAP_NATIVE_EXECUTION_20261103.md)。

**22:01执行授权：** 用户确认先按固定LEAP原配方完成原生验证，再做Sharpa；不得自行缩减网络/训练规模，短试跑后报告并等待正式长训确认。目标午夜前具备启动条件；若Sharpa未验收，不能用LEAP结果冒充。只维护一套参考学习器和必要适配，旧实现/环境/历史记录保留。见 [原生执行计划](LEAP_NATIVE_EXECUTION_20261103.md)。


**21:53最新审查：** 用户质疑训练配方偏离参考。已确认环境/batch/网络、状态积分、归一化和失败学习存在未充分验证的差异；此前参数缩放不能证明复现有效。建议以固定版本LEAP学习器为基础，先原生验证再做Sharpa适配，Wuji作为参考动作实现参照。四动作方向保留，原扩容/长训数值暂停定案；新训练未启动，现有筛选不改。见 [参考偏离审查与迁移建议](REFERENCE_BASELINE_AUDIT_20261103.md)。

**11月3日训练问题离线审计：** 只读取两个已保存checkpoint并在CPU上做固定记录动作展开，无参数更新、无CUDA、无新训练。确认当前损失按状态标准差加权，物体位置的该尺度约为增量尺度10倍；原版抽样H8损失约59.9%来自外力，候选H8物体姿态平均误差约14.76°。两者使用各自内部回放，不能据此比较模型优劣或证明唯一根因。结合既有失败监督隔离/预热结果，当前证据优先指向训练路径、数据覆盖和模型表示/损失，模型宽度与激活函数的独立影响仍未测。[审计细节](SM_REPRODUCTION.md#11月3日训练问题离线审计) · [机器证据](../work/analysis/sm-failure-analysis-20261103/checkpoint-audit.json)。

**最新方向已同步：** 用户要求单物体不同动作优先、多物体后移，并提高训练与实验迭代速度。已审查当前接口：缺少时变参考，持续旋转的stall终止与保持动作冲突；动作、外部单物体子集及加速方案已更新。新动作/进一步融合与并行评测均待实现；本次只改文档和提案，不启动新训练。

**外部benchmark审查：** 已核对SM训练动作覆盖口径及LEAP手部SM、DexTrack、AdaDexTrack、DexMimic等作者报告；当前Sharpa适配仍锁定单物体，未实现多物体几何条件或多轨迹目标。外部benchmark尚未运行，不能视为已验收。用户要求与分步计划已记入[外部验证文档](EXTERNAL_BENCHMARKS_20261103.md)。本次未改训练源码或启动新任务。

**资源优化追加：** 用户要求提高并行/显存利用率。短测显示512环境采集吞吐约为64环境的5.3倍；batch2048的更新单元耗时接近256。合并有限性检查在CPU及GPU测试中保持更新一致。已在本项目三卡各并行一个最多8分钟的完整循环容量试跑，原筛选61项源码与预算不变；三档24轮完整循环均已通过；512档样本吞吐约提高6.6倍、actor更新/小时减少约18%。用户新增24GiB和计算密度目标，见[后续提案](GPU_24GB_PLAN_20261103.md)；更大档尚未启动。[实测、影响和配置方案](GPU_THROUGHPUT_20261103.md)。

**11月3日晚最新：** 原1/3号完成最终200回合，回报613.75/748.84、旋转30.63/49.70 rad、失败6%/4%；3号达到50%联合门槛，未达80%。八组分支/恢复试跑已通过，19:05正式启动联合筛选。GPU0被其他用户占用，使用空闲1/2/3号，按3＋3＋2排队、每组最多80分钟；预计22:30—23:05完成，23:20硬截止。筛选后先报告再确认夜间从头长训。[验收与今晚计划](SM_SCREENING_20261103.md)。以下为此前快照，不能作为当前运行状态。

**Current objective: a strong SM learning baseline on the Sharpa hand, followed by WSM policy-context augmentation. The complete loop passed engineering pilots; modified SM candidates show rotation learning on development evaluations, with final tests and stable precision benefits still pending. Original SM/PPO seed 92001 controls have completed; two isolated SM candidates are still running at the latest check.** See [MAINLINE.md](MAINLINE.md) for accepted scope and [ROADMAP.md](ROADMAP.md) for deliveries.

Documentation snapshot: **3 November 2026 (Asia/Shanghai)**. The runtime check completed at `2026-10-31T17:38:52Z`, or 1 November, 01:38 in Shanghai. This is an evidence record, not a live job monitor.

## Latest approval change and experiment hold (10:08 Shanghai)

All four new pilots passed. The earlier authorized gate launched A00/B10 at 08:59; after the user's latest requirement for formal confirmation before long training, both were gracefully stopped at 600 cycles and their finite checkpoints preserved. C01/D11 queues were canceled before training. New-run GPU use was 2.2921 hours plus 0.3594 pilot hours = 2.6515 GPU-hours. Automatic authorization is revoked; no new long run or resume without the user's formal confirmation. Original GPU1/3 candidates remain running under their existing deadline. [Stage results, audit limits and revised proposal](BASELINE_REVIEW_20261103.md). The execution notes below describe the earlier snapshots.

## Current priority: a strong Sharpa SM baseline

The user explicitly prioritizes reference implementation review, detailed Sharpa adaptation audits and measured configuration optimization before the SM+WSM comparison. [Concrete tasks and review checklist](SM_REPRODUCTION.md) now define reference selection, source/config differences, diagnostics, controlled tuning and a frozen baseline handoff. Both future comparison arms must inherit baseline fixes. The original GPU0 implementation remains a historical diagnostic control, not the intended main baseline.

Reference screening covered four public repositories, with different review depths documented in the [stage report](BASELINE_REVIEW_20261103.md). The implementation adds an isolated 2×2 baseline experiment: importance-corrected failure sampling × real-only critic normalization, with curriculum persistence repaired in all four arms. Eleven distinct CPU checks and all four bounded real pilots passed. The formerly authorized automatic long-run gate is now revoked as described above. Existing hashed candidate sources remain unchanged. Learning benefit from the new changes and the final strong baseline are still pending. [Historical experiment plan and resource/information audit](SM_BASELINE_EXPERIMENTS_20261103.md).

At the earlier 08:54 snapshot, the first pilot pair A00/B10 had passed actual 10+2 resumed cycles and 8-episode physical checks (320 model / 12 actor / 240 critic updates each, 4.78/4.71 seconds per cycle). C01/D11 subsequently passed the same checks. All four short policies failed all 8 episodes; this is engineering evidence only.

## 3 November result check

GPU0 original SM and GPU2 PPO completed their 200-episode locked tests and exited normally by 07:19 Shanghai time. SM: return 330.62, rotation 1.36 rad, 48/200 failures. PPO: return 762.38, rotation 56.54 rad, 7/200 failures. PPO learned sustained rotation; original SM still lacks effective rotation despite increased return/survival. Best/last checkpoints were verified, and final evaluation hashes match the selected best weights.

At approximately 08:04, GPU1/3 candidate processes and W&B runs were active. Latest 32-episode development returns were 600.23/734.57, with 7/32 and 1/32 failures; final tests are pending until the original 11:19–11:49 window. No fatal candidate error was found in inspected logs; the user's specific bedtime error remains unidentified. [Result and status report](ONLINE_RESULTS_20261103.md) · [Evidence](evidence/online-results-20261103.json).

## Run allocation at launch (2 November 23:49)

At 23:49 Shanghai time on 2 November, the user-authorized candidate runs launched on GPUs 1/3. GPU0 original SM and GPU2 effective PPO seed 92001 remain running with unchanged sources/deadlines. The two seed-92002 jobs were cleanly stopped with their checkpoints preserved. Candidates test detached termination supervision, then additional online model warm-up; their control benefit is not yet known. [Risk, acceptance and commands](SM_CANDIDATE_RUN.md).

## Current reproduction status (earlier 23:27 snapshot)

**SM reproduction remains incomplete.** The current conversation prioritizes successful from-scratch learning; the user clarified that formal document deliverables remain unchanged. At approximately 23:27 Shanghai time on 2 November, all four lanes had run 3.87 hours without process/numerical failures. Latest 32-episode development results: SM seeds 92001/92002 returned 6.94/15.71 with 32/32 failures each; PPO seeds 92001/92002 returned 656.76/2.28 with 1/32 and 32/32 failures. PPO-92001 met the existing 50% capability milestone, providing partial evidence that the task is learnable; SM has not. These are interim development results, not final tests.

A CPU-only audit of saved checkpoints and additional community implementations found differences in model warm-up, termination sampling/gradient coupling and critic normalization that require controlled diagnosis; no unique root cause is established. [Reproduction criteria and debug tasks](SM_REPRODUCTION.md) · [Progress, checkpoint and source evidence](evidence/sm-reproduction-audit-20261102.json). Active code/config/checkpoints were preserved.

## Current versus historical evidence

W&B authentication, target-project writes and remote readback of both pilots are verified. Entity: `lyrine-huazhong-university-of-science-and-technology`; project: `unionmimic`; SDK: 0.22.3. Each algorithm/seed has its own stable tracking ID. Local JSONL/TensorBoard remain available. The old two-hour watchdog stays disabled.

The earlier [key-format fix](evidence/wandb-key-fix-20261102.json) and [offline check](evidence/wandb-setup-20261102.json) are historical setup evidence. Real authentication and online pilot logging were verified after the user completed login; current records are in `work/analysis/online-sm-20261102/wandb-verification.json`.

New baseline acceptance: **13 numerical, gradient, boundary, resume, tracking and orchestration checks passed**. SM completed 150 cycles (1,700 model / 150 actor / 3,000 critic updates); PPO completed 90 cycles (1,440 actor/critic updates). Both resumed in real processes. Independent 16-episode evaluations all failed: random initial return 4.16, SM −4.81, PPO −13.84. Engineering acceptance does not establish control benefit. [Evidence](evidence/online-sm-pilots-20261102.json) · [Current plan and evaluation protocol](ONLINE_TRAINING_PLAN.md)

| Item | Evidence state |
| --- | --- |
| Original physics/sensors, aligned data, RSSM recovery | Verified on the legacy path; records below |
| Legacy actor BPTT and PPO/RSSM adaptation checks | Existing numerical/boundary/resume evidence; does not verify a separate SM model |
| SM from-scratch online loop | Engineering verified; learning benefit unproven |
| Frozen WSM context and imagined observation bridge | Interface defined; implementation/acceptance pending |
| SM vs SM+prior precision/cost comparison | No result |

`adaptation-20261102-v1` launched on 2 November with two PPO fine-tuning lanes and two RSSM-fitting lanes, then stopped at the user's direction at 17:28 Asia/Shanghai. Latest complete 32-episode development returns were −0.24%/−1.66% for PPO and −1.04%/−2.62% for RSSM versus the original reference. These evaluations precede the final saved checkpoints; no locked final test was run. Nine checkpoints deserialize with finite saved tensors, all four training process trees exited, and original weights remain unchanged. All 33 launch source hashes matched at the stop check; the archived source snapshot remains intact. A subsequent workspace change to `scripts/env.sh` for W&B configuration was preserved; the stopped jobs were not resumed. [Stop evidence](evidence/adaptation-20261102-stop.json)

The old run used 7.942 GPU-hours; including 6.059 preflight hours, 33.999 remain from the 48 GPU-hour envelope. It provides engineering/diagnostic evidence, not a test of SM from-scratch learning or the new prior comparison. That was the balance at the historical stop. New pilots consumed 0.943 GPU-hours. The latest user instruction supersedes the cumulative 48-hour cap and authorizes four idle GPUs for up to 12 hours each; the current run uses two SM and two PPO seeds, with evaluation inside each deadline.

## Historical runtime acceptance

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

- Complete SM from-scratch learning, its PPO-Sim reference, and SM/SM+prior precision/cost curves.
- Cross-object, cross-embodiment or real-robot transfer.

Promote a proposed capability to a result only when its evaluation artifacts are available.

## Data and first training run

On 1 November, the next acceptance run collected 99 complete episodes (24,659 transitions), verified exact action/target alignment and RSSM posterior recovery, and passed H=1/H=4 prediction checks. It completed 2,500 prediction-head updates, 200 actor updates and a 25-update checkpoint resume. Four regression tests passed.

The H=4 reward RMSE was 2.4285 versus a constant baseline of 2.7687; normalized joint RMSE was 0.1040 versus persistence at 0.4815. Failure recall was 75% with 6.08% false positives on 12 held-out failures. Failure probabilities still need calibration: their Brier score did not beat the constant baseline.

Physical evaluation used seed 123 and eight complete episodes per policy: the original mean return was 1073.56 and the updated policy returned 1019.41, a 5.04% decrease. Neither policy failed. This verifies execution, not a performance gain.

A bounded overnight loop was launched and subsequently replaced, at the user's request, by four independent seeds under tmux with a 12-hour budget per GPU. Live states are in `work/runs/four-gpu-20261101/gpuN/status.json`; local TensorBoard is enabled, W&B sync is disabled for this first run. See [training controls and estimates](TRAINING.md), [acceptance details](STEP2.md), and the [machine-readable evidence](evidence/step2-summary.json).

## Final four-card result

The run ended by 1 November, 14:44 Asia/Shanghai. All four training processes have exited; the two-hour watchdog was canceled at the user's request. TensorBoard remains available. Final totals: 117,875 attempted SM updates, 232,000 head updates, 1,876,649 new complete transitions and 235 evaluations (1,880 episodes, all on the reused development seed).

Relative to each lane's initial evaluation, the last ten evaluations averaged −0.27%, −1.77%, +0.53%, and −3.09%. Historical best checkpoints reached +0.53% to +4.96%, but repeated selection on small evaluations is not evidence of a reproducible gain. A new-seed, eight-episode probe measured 1,091.38 for the initial policy and 1,099.28 for GPU2's saved best (+0.72%); this is too small a test for a firm conclusion.

Final H=4 reward RMSE was 1.756–1.797 versus constant baselines 2.081–2.088, and normalized joint RMSE 0.092–0.095 versus persistence 0.495–0.500. The effective validation failure positives still came from the original 12 failure episodes; thresholds and head selection used that validation data. These checks do not measure action-gradient fidelity.

Final evidence and figures: `work/analysis/first-run-final/report.html`, `final-summary.json`, `data-audit.json`, and the reward/gradient probe JSON files. The [historical follow-up protocol](EXPERIMENTS.md#first-run-follow-up-protocol--2026-11-01) records the subsequent fine-tuning design. The later PPO/RSSM run is distinct from the still-pending SM-versus-prior comparison.
