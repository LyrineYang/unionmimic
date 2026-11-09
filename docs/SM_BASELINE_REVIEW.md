# 从随机策略开始的 SM 闭环复查

参考版本：第三方手部学习器 `in-hand-rotation-mjlab@0ab03eb2244b4a1a020d94ef38a4823bb9e15ba0`。它是算法适配参考，不是官方 SuperMimic 发布。环境与 PPO 损失来自 `WM-Craftnet@a1e825acaf50ade298e30b0b13c6671627af3a1e`，沿用本仓库已记录的环境修补。

**复查范围说明：** 以下是第一轮实现契约复查，不是端到端学习正确性的验收。新增参考源码与断点诊断已发现需验证的预热、失败采样/梯度耦合、critic等差异；详见 [SM_REPRODUCTION.md](SM_REPRODUCTION.md)。SM复现当前未完成。

11月3日后续重点已明确为：对照参考源码和配置审查Sharpa适配，优化并固定强SM基线。下表是第一轮历史实现说明，不是所有差异已经解决的清单；具体任务、参考筛选和验收依据维护在上述复现文档。

## 实现对应

| 参考工作 | 当前实现 | 验证/适配说明 |
| --- | --- | --- |
| `train_yaw.py` 在线采集与 GPU Replay | `online/simulator.py`、`core.Replay` | 当前 actor 实际用于每轮采集；22维命令、执行动作及目标更新使用原任务。保留四帧292维观测 |
| k步记录动作自由预测 | `learner.fit`、`task.PhysicalModel` | 独立177维物理/控制器状态、93维上下文；k=8。预测状态自回归，不注入未来真实状态；动作噪声和scale使用记录输入 |
| 模型预热和在线重拟合 | `worker.py` | 随机 actor/critic/F；128个向量步预热、200次模型更新，再交替采集8步和拟合10次；无发布 actor、WSM或教师回放 |
| 冻结模型，BPTT策略 | `learner.imagine` | H=8，固定F/critic参数但保留输入导数；折扣奖励加尾值，actor梯度norm1再value0.1 |
| TD(λ) critic | `core.lambda_returns`、`learner.imagine` | 与固定参考源码逐值对照；分离图，5×4个critic minibatch。超时在发生点bootstrap，后续想象不计入目标 |
| PPO-Sim | `learner.ppo` | 直接调用固定上游 `actor_loss`、`critic_loss` 和 adaptive-KL scheduler；新建共同actor/critic，保存原始Gaussian样本用于概率比，裁剪后命令用于仿真 |
| 断点与恢复 | `worker.py` | actor/critic/F、四组optimizer、normalizer、RNG、GPU replay及计数联合保存；额外保存物理/控制器快照 |

## 与参考不同、需要明确报告的部分

- 平台是 Sharpa/Isaac Gym/当前z轴物体；奖励、控制频率、传感器与参考 MuJoCo 手部任务不同。不是 LAFAN1 数值复现。
- F采用残差MLP并约束四元数归一化；控制器解析推进。物理预测覆盖关节/速度、物体位姿速度、指尖位置、接触力、力矩及有限差分速度、外部力。原奖励各项根据这些量计算；接触阈值保持原语义，失败分类器负责想象的可微存活权重。
- 原环境有随机外力。当前力在状态内，质量/摩擦/增益/重力/动作相关噪声在上下文内；未来随机外力创新未知，确定性F学习其条件预测，不能声称逐轨迹精确重建该随机过程。
- Actor均为512–256–128，critic为256–128；SM使用有界确定性输出加探索噪声，PPO使用同一均值网络加可学习Gaussian尺度。该PPO适配复用原损失及调度器，不声称原始rl_games训练器逐字复现。
- 数据按episode ID分训练/验证，F与归一化统计不使用验证episode；失败和超时不丢弃。训练完成回合曲线含探索噪声，须与独立确定性开发评测分开。
- 跨进程不能恢复PhysX内部接触求解器缓存。恢复精确保留learner/replay/RNG，再以新episode ID开始仿真；记录被截断回合，不把重置接成旧轨迹，也不声称物理状态逐位连续。
- 前置CPU检查包含真实回放上的F更新、冻结F后的actor/critic更新，以及相同联合断点/RNG的下一次更新一致性。仿真短试跑、实际进程恢复和学习能力是后续独立证据。

## 验收入口

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_online_learning -v
```

其中两个真实回放检查需要已生成的 `work/runs/online-sm-20261102-probe02/last.pt`。该依赖是有界仿真产生的工程fixture，不是专家训练数据；正式训练不会加载它。

实测状态、短试跑结果及长训决策统一记入 [TRAINING.md](TRAINING.md) 和 [STATUS.md](STATUS.md)。本复查不能单独证明物理策略有效、达到预期精度或比PPO更快。
