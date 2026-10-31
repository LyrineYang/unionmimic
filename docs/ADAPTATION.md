# 第 2、3 步：RSSM 与 SM 适配

依据：WM-Craftnet `a1e825a`、发布的 z 轴 checkpoint，以及 LEAP SM 参考 `0ab03eb` 的 `scripts/wm/train_yaw.py`。这里描述第 2、3 步的实施方案，尚未开始 SM 训练。

## 可复用与必改

| 部件 | 直接复用 | 必须补充或修改 |
|---|---|---|
| 感知与 RSSM | 深度预处理、encoder、`obs_step`、`img_step`、原权重 | 暴露完整循环状态 `deter/stoch/logit`，不能只存交给 PPO 的 512 维特征 |
| 原策略 | WSM 特征投影、策略 MLP、22 维动作输出及输入归一化统计 | 增加可求导的 actor 入口；绕过 `PpoPlayerContinuous.get_action()` 和 `BaseModelNetwork.norm_obs()` 内的 `no_grad`，冻结统计但保留输入梯度 |
| 预测头 | 原有 prop/depth decoder、奖励头结构 | 校准奖励头，启用非零奖励损失；新增失败/存活预测头；训练并评估先验展开时的预测 |
| 控制与任务 | 原动作加噪、平滑、缩放、限幅、目标累积、奖励、重置、物理资产 | 保存控制器状态，在想象中复现动作映射并重建观测历史 |
| 优化器与价值网络 | LEAP 的短段目标、存活权重和 TD(lambda) 算法逻辑 | 新建 SM 优化器与可用潜状态作为输入的 critic；不直接沿用读取特权物理状态的 PPO central critic |

原策略的输入是四帧低维观测加 RSSM 特征，不直接读取深度图。单帧为 73 维：22 关节位置、22 目标位置、5 接触值、24 轴方向编码；四帧为 292 维。已核对发布 checkpoint：归一化统计为 292 维，actor 首层输入为 324 维（292 + RSSM 投影 32），输出为 22 维。

## 执行顺序

| 顺序 | 修改 | 交付物 | 验收 |
|---|---|---|---|
| 2.1 采集对齐数据 | 扩展原采样/回放，保存观测、策略命令、执行动作、控制器状态、奖励、失败、超时、episode ID；记录重置前的终止观测 | 原策略生成的数据及校验报告 | 检查 `o_t → a_t → r_(t+1), o_(t+1)` 对齐；不把重置后的观测当作动作结果；验证集按完整 episode 划分 |
| 2.2 恢复预测起点 | 用真实历史运行 `obs_step`，提取完整 RSSM 状态、四帧策略历史及 `prev_targets/last_actions` | 可重建的想象起点 | 相同历史与随机状态可复现结果；模型更新后重算起点；起点前使用历史预热，不把任意中间帧当新 episode |
| 2.3 补齐短段预测 | 先冻结感知主体，训练奖励和失败头；调用 `img_step` 展开 4 步，必要时微调 dynamics 和 prop decoder | 校准 checkpoint、1/4 步预测报告 | 保留成功与掉落样本；报告奖励误差、失败召回及误报；超时单独处理；先验预测优于常数/保持上一帧参照，再接策略训练 |
| 3.1 接回原 actor | 关节和接触值取 prop 预测，轴方向保持已知值，目标位置按原控制公式更新，组成下一帧并滚动四帧历史；保留原归一化和动作限幅 | 可求导 actor 与观测适配器 | 输入真实观测时，动作与原 player 的确定性动作一致；归一化统计不变，观测到动作的梯度存在；不用未来真实观测 |
| 3.2 实现 SM 更新 | 从 2.2 的状态开始，actor 产生动作，RSSM 预测下一步，累计奖励及末端价值；固定模型/critic 参数更新 actor，再独立拟合 critic | 4 步 SM learner | actor 参数确实改变；actor 更新期间模型和 critic 参数不变；梯度有限；分离末端失败与超时的自举处理 |
| 3.3 连接在线循环 | 当前策略回原仿真器采样，更新模型后再进行短段 SM 更新；保存全部训练状态 | 可持续运行和恢复的 WM+SM 入口 | 完成数百次短更新、保存恢复及固定物理评估；记录模型内回报与仿真回报，不能只凭想象回报判定改善 |

SM 目标：`J = sum(gamma^t * survival_t * reward_(t+1)) + gamma^H * survival_H * V(state_H)`。首轮 H=4（原控制约 10 Hz，即 0.4 秒）；reward/critic 使用一致量纲，单独记录 PPO 的原始奖励与缩放奖励。

动作适配必须区分策略命令和 RSSM 输入。当前相对控制先对命令加噪、限幅，再计算 `a_t = 0.8 * command_t + 0.2 * last_actions`；RSSM 接收这个 `a_t`。目标关节位置按 `clip(prev_targets + scale_t * a_t)` 更新。3.1 需要复现该映射：固定随机样本时与仿真器逐步对齐，训练时按原分布采样噪声和比例。actor 首版使用均值输出。

3.2 冻结模型和 critic 的参数，但保留其对输入的梯度；想象前向不能放进 `no_grad`。SM learner 使用 WM-Craftnet 的 RSSM，LEAP 只提供更新逻辑参考。

## 关键接口位置

- `isaacgymenvs/utils/world_model.py`：状态缓存、回放、模型训练/保存；原 `step()` 保持推理用途，新增独立想象入口。
- `wsm/networks.py`：复用 `RSSM.obs_step/img_step/get_feat`；保留离散潜变量的 straight-through 梯度路径，不直接改成不可导 argmax。
- `wsm/models.py`、`wsm/configs.yaml`：当前奖励损失默认 0，继续执行头被注释；新增头不影响原版 checkpoint 的严格兼容恢复。
- `rl_games/algos_torch/network_builder.py`：原 actor 拼接低维观测和 `wm_feature`；复用该网络，另写有梯度的归一化入口。
- `isaacgymenvs/tasks/wm_craftnet_camera_rotation.py`：真实采样持续使用深度和触觉修正 RSSM；想象阶段不渲染、不使用未来传感器观测。
- `isaacgymenvs/learning/`：新增 SM learner，保留原 PPO 入口；从 LEAP 参考中抽取更新逻辑，不引入其 MuJoCo 依赖。

发布 z 轴权重对应 prop/depth 辅助头。先按匹配配置加载，再显式初始化新增失败头；不使用宽泛的 `strict=False` 隐藏已有模块不匹配。
