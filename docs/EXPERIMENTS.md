# Experiment design

**2026-11-09 重新授权：** 本次24小时窗口固定至11月10日19:44:55（上海），包括修复、验收与训练；全部验收通过后自动启动四卡SM/PPO，无需再次确认。新候选整段相位首步通过354/416（85.1%），尚未达到90%，正修复接触与重定向；长训尚未启动。[现场与计划](PEN_ADAPTATION_AND_QUEUE_20261109.md) · [可展示demo](DEMOS.md)。


**11月4日最新决定：完整Sharpa多动作适配优先。** 角块SM/PPO长训已于11:28:50停止，权重和历史评估保留，不自动恢复。50Hz、四任务物理接口与模拟位姿观测已有短测；单笔抓取未过验收，完整参考命令、学习器/想象接口仍待完成。先完成这些适配，再比较SM/PPO，之后接WM-Craftnet。[验收缺口与执行顺序](SHARPA_FULL_ADAPTATION_20261104.md)。以下日期记录属于历史快照。


**21:53最新审查：** 用户质疑训练配方偏离参考。已确认环境/batch/网络、状态积分、归一化和失败学习存在未充分验证的差异；此前参数缩放不能证明复现有效。建议以固定版本LEAP学习器为基础，先原生验证再做Sharpa适配，Wuji作为参考动作实现参照。四动作方向保留，原扩容/长训数值暂停定案；新训练未启动，现有筛选不改。见 [参考偏离审查与迁移建议](REFERENCE_BASELINE_AUDIT_20261103.md)。

**最新顺序：** 用户明确先单物体不同动作，暂缓多物体；外部单物体动作子集和本地指令诊断分开报告。训练速度与实验周转分别验收；当前旧实验协议不变，新增动作尚未实现。详见 [动作计划](EXTERNAL_BENCHMARKS_20261103.md) 与 [资源计划](GPU_24GB_PLAN_20261103.md)。

**11月3日外部benchmark要求：** 强baseline需增加固定外部任务集、成功条件及逐任务结果；当前内部回报和能力门槛不能独立支撑泛化结论。参考数据、报告数字、权限审计和多动作适配计划见 [EXTERNAL_BENCHMARKS_20261103.md](EXTERNAL_BENCHMARKS_20261103.md)。外部实验仍为提案，现有筛选协议不变。

**11月3日晚最新：** 原1/3号完成最终200回合，回报613.75/748.84、旋转30.63/49.70 rad、失败6%/4%；3号达到50%联合门槛，未达80%。八组分支/恢复试跑已通过，19:05正式启动联合筛选。GPU0被其他用户占用，使用空闲1/2/3号，按3＋3＋2排队、每组最多80分钟；预计22:30—23:05完成，23:20硬截止。筛选后先报告再确认夜间从头长训。[验收与今晚计划](SM_SCREENING_20261103.md)。以下为此前快照，不能作为当前运行状态。

目标与范围以 [MAINLINE.md](MAINLINE.md) 为准；本文定义实验对照与验收。H1=策略上下文收益，H2=真实/想象上下文一致性，H3=上下文辅助物理动力学，不能互相替代结论。

## 当前主实验：以 SM 为主体，加入 WM-Craft 状态先验（2026-11-02）

用户已确认：**继续当前灵巧手任务，先实现从随机策略开始的 SM 完整训练闭环，再比较加入 WM-Craft 先验的效果。** 目标是改善训练后策略的控制精度，同时记录训练成本。以下方案替代后文以发布策略微调为主的实验设计；历史结果及当时的验收条件保留。

### 当前执行状态与任务顺序

**最新覆盖：11月3日10:08，用户要求正式确认后才启动长训。** 新A/B已正常停止，C/D排队取消，原1/3号继续；下面四组自动启动安排已失效。后续推荐先做短分支联合筛选，再报告选定的从头长训配置，见 [阶段报告](BASELINE_REVIEW_20261103.md)。尚未确认或启动新的筛选/长训。

11月3日新增执行：[baseline四组交叉实验](SM_BASELINE_EXPERIMENTS_20261103.md)。基于原3号配方，共用课程恢复修复，比较失败重要性采样与critic仅真实数据归一化的主效应和交互；有界试跑通过后四组各9小时。预算、特权信息、原论文配置差异和后续验证要求均随计划记录。

2026-11-02 17:28（上海时间），按用户授权停止 `adaptation-20261102-v1`。最近完整开发评估没有控制收益，RSSM 的 H4 奖励/关节 RMSE 改善不足1%；现有数据和断点保留作工程参考。原48 GPU小时额度已使用14.001，剩余33.999。详见 [停止证据](evidence/adaptation-20261102-stop.json)。

**T0–T2完成了工程检查，尚未完成强SM基线验收。** 11月3日原GPU0 SM / GPU2 PPO已完成；PPO学会持续旋转，原SM未学会。GPU1/3修正候选已有有效旋转的中期证据，最终测试仍待完成，见 [结果报告](ONLINE_RESULTS_20261103.md)。用户明确当前优先工作为参考复现库的源码/设置核查、Sharpa适配审查和有序调优，具体执行见 [SM_REPRODUCTION.md](SM_REPRODUCTION.md)。先固定足够强的SM，再进行SM+WSM主比较；两组共享基础修正，不能以增强版对旧0号或仅对WM-Craft/PPO的收益替代主比较。以下T0–T5交付职责及既有数值门槛保留。

资源历史：用户授权四张空闲卡各最长12小时；原方案为SM/PPO各两个配对种子，后按授权将两条92002任务替换为SM候选。每卡11.5小时训练/开发评测＋30分钟锁定测试。原累计48 GPU小时不再是硬上限，历史消耗仍保留。原配置与候选配置分别见 [ONLINE_TRAINING_PLAN.md](ONLINE_TRAINING_PLAN.md) 和 [SM_CANDIDATE_RUN.md](SM_CANDIDATE_RUN.md)；下面的试跑候选参数不代表已调优配置。

| 任务 | 具体工作 | 输出 | 通过条件 |
| --- | --- | --- | --- |
| T0：任务与评测适配 | 固定当前物体、轴、reward、终止、随机化和控制器；记录物理状态/上下文、实际传感器、真实执行动作；补轴偏差与非目标轴运动 | 版本化字段/配置说明，数据对齐报告，独立开发/测试清单 | 原环境与适配器动作、reward、终止逐项一致；失败/超时/reset首帧不混淆；评测不只统计幸存回合 |
| T1：独立 SM learner | 新建物理预测模型、随机 actor/critic、GPU replay；当前策略采集→记录动作拟合 F→短时想象 BPTT→TD(λ) critic | 可持续运行的 learner、模型/策略/优化器联合 checkpoint、每阶段计数和耗时 | 至少20个完整交替循环；actor变化实际进入后续采集；F和actor各自更新，actor阶段冻结F但梯度可穿过；有限值与保存恢复检查通过 |
| T2：两条有界试跑 | 同一任务、同一随机 actor 起点，分别试跑 SM 与原生 PPO-Sim；各最多1单卡小时 | 每条真实回报/旋转/失败曲线、吞吐/显存、模型与策略更新证据、恢复后的实际继续更新 | 跑通且数据/梯度/断点契约成立；分别记录有无学习趋势，不要求随机策略达到发布策略95%水平 |
| T3：第一阶段学习对照 | T2已通过；SM与PPO-Sim各2卡，两个配对种子，每卡12小时含评估 | 从随机起点的学习曲线、物理精度、达到固定能力的成本及最终评测 | 检验 SM 是否学到控制以及成本特点；分别报告两个训练种子，不把评估回合数冒充训练种子数 |
| T4：冻结先验接口 | T1可用后接入原 WSM、ψ、预测观测/encoder特征桥接；基于记录动作恢复并递推上下文 | 真实与想象上下文/动作差异、无未来泄漏证明、冻结与梯度检查、额外开销 | 满足目标观测条件更新接口；action-only prior 仅作为单独标注候选，不冒充等价实现 |
| T5：主比较 | 基础SM与SM+prior，每组2个配对训练种子 | 精度—成本曲线、逐种子结果与最佳 checkpoint | 固定任务和目标；保持旋转/失败约束；分开解释额外信息/容量与预训练贡献 |

T0/T1 优先在独立模块中实现，保留旧训练入口作为历史复现路径。T1 应常驻仿真和回放，避免每轮重新加载仿真器、导出完整数据再重建全部缓存；吞吐提升必须以实际计时报告。旧教师轨迹不默认填入新 SM 的训练回放，先验资产、调试数据与当前策略训练数据分别标记。

**T2 起始配置（候选，正式训练前按验收与实测固定）：**

| 项目 | 配置与目的 |
| --- | --- |
| 观测/动作 | 沿用当前可部署本体/接触/目标历史及22维相对命令；两算法共用 actor 结构和基础参数初始化，不加载发布 actor/critic，也不使用其归一化统计或动作 anchor |
| Actor / critic | 候选 actor MLP 512–256–128、critic 256–128；两组保持相同信息和容量。critic可读物理训练状态，actor不可读取特权状态 |
| F | 独立自回归物理状态模型，候选512–512–512；先明确覆盖原reward/终止/观测所需字段。可解析项复用任务定义，学习项明确列出，不偷偷重写奖励 |
| 并行采样 | 先64环境检查，资源允许后128；SM与PPO使用相同起始配置，正式阶段按测时冻结；不从H100性能外推V100 |
| 模型拟合 | 窗口k=8，batch256，候选每次10个模型 minibatch；记录实际物理转移、采样窗口、模型/actor更新数和比例 |
| 想象/BPTT | H=4用于接口检查，H=8约0.8秒作为首个训练候选，batch256；actor/模型候选LR=2e-4，actor梯度norm1再value0.1；γ=.99、λ=.95，critic每次5×4 minibatch |
| 模型预热 | 先采集至少128个向量环境步并完成200次模型更新；所有时间计费。数值/数据检查通过后进入actor交替更新，不以旧冻结RSSM方向门槛永久阻断新模型学习 |
| PPO | 复用上游PPO损失与其从头学习配置，适配共同actor/critic和观测；不沿用强策略微调专用小学习率。Gaussian采样与SM探索机制的差异明确记录 |
| 验证 | 固定开发集周期评测，独立200回合测试在选定checkpoint后打开；每步物理量用于精度，不把loss或episode reward单独当精度 |

T3的初始能力里程碑设为：回报与有符号旋转量**同时**达到同条件发布策略的50%、80%，且该评测批次失败率≤5%；报告失败率区间，不能把单批点估计当成非劣性证明。发布策略只用于测量参照，不提供训练动作或经验回放。T0先核对参照条件，正式训练前写入评测清单；若参照条件改变则重测，不在比较结果出来后移动门槛。未达到就报告预算内未达到；只有两组达到同一里程碑时计算速度比。最终控制精度仍单独报告轴偏差与非目标轴运动。

这些是用于试跑的具体起点，不声称与论文平台/超参数完全相同。H8的物理时长和独立F的误差须实测；若调整，修改正式配置并记录原因，不在组间比较后调验收目标。

**试跑前预算设计（已被最新四卡12小时授权替代，保留决策历史）：** T0/T1首先做CPU数值/数据检查；T2合计最多2 GPU小时。T3训练占16 GPU小时，开发评测计在各自8小时窗口内，最终评估另预留最多4 GPU小时，下一阶段合计上限22 GPU小时，低于当前剩余33.999。该上限是任务设计，不是已经启动的资源占用。数值/数据错误及时停止相关任务；健康T3按8小时执行，用户的新停止指示优先。

设备候选：GPU0跑SM、GPU2跑PPO；GPU1/3分担短试跑与最终评估，各最多3小时。前置2 GPU小时同时覆盖T0/T1必要的GPU验证，不能额外漏计。按已用时间累加，该分配各卡仍低于原12小时累计上限；启动时重新检查可用性，不停止或挤占其他项目。若改用其他分配，须同时核对单卡与总账。

T4使用扣除实际消耗后的余额，先测接口开销。T5的四卡×8小时仅训练就需32 GPU小时，不能在T3之后假设原额度仍足够；在代码和测时齐备后另列可执行预算，或由用户明确重分配，不能自动再开一轮48 GPU小时。若T2不通过，保留原因并修复，不启动T3；T3未观察到学习时，优先诊断闭环和奖励/模型，不用增加WSM掩盖基线问题。

### 1. 问题与适用范围

主问题：在相同灵巧手、物体、旋转任务和训练预算下，SM 策略增加 WM-Craft 的历史状态输入，能否更准确地维持旋转轴并减少失稳，同时保持旋转进度？

这次复现 SuperMimic 的学习循环和控制实验设计。平台仍是 Sharpa 灵巧手、Isaac Gym、单物体绕 z 轴旋转；不称作 G1/LAFAN1 数值复现。论文的 0.82 回报和 1.2 小时不能直接作为本任务的门槛或耗时预测。

论文明确每轮采集当前策略数据、更新预测模型、再通过想象轨迹更新 actor/critic。主策略随机初始化的具体分布没有在所核对的算法中完整列出，本项目将单独固定并记录初始化方案。原 SM 已使用观测历史，不能用“单帧策略”替代其基线。[SuperMimic，§3、图4、算法A.1、表A.1/A.5](https://arxiv.org/pdf/2610.28480)

### 2. 两个模型分别负责什么

| 模块 | 职责 | 第一版训练方式 |
| --- | --- | --- |
| SM 动力学模型 F | 从物理状态、执行动作和仿真上下文预测后续状态，供想象和 BPTT 使用 | 随机初始化，持续用当前策略采集的数据拟合；两组采用相同结构、监督及更新规则 |
| WM-Craft 状态模型 Q | 从深度、本体/触觉观测与历史执行动作估计历史状态 | 加载同一份发布先验，第一版冻结其参数；真实交互时持续更新内部状态 |
| 投影 ψ 与 actor π | 把历史状态转换为策略可用的额外特征，并产生动作 | 从头训练；不加载发布 actor，也不约束动作靠近发布 actor |
| critic V | 对短时想象之后的收益提供估计 | 从头训练，两组使用相同架构、状态信息与更新规则 |

策略接口为：

```text
SM:       a_t = πθ(o_history,t, g_t)
SM+prior: a_t = πθ(o_history,t, g_t, ψ(h_t))
```

其中 g_t 是当前任务的目标旋转轴/方向；现有任务没有逐时刻参考关节动作或目标转角轨迹。两组保留相同的观测历史长度。WM-Craft 的先验是否“精确”、策略是否会利用它，均是要测量的结果。

先验不替代 F 的状态转移、奖励或终止模型，也不直接作为动作教师。为减少混杂，第一版不让 critic 额外读取 h，不同时开展 WSM 联合微调。

### 3. 从采集到下一轮策略的完整闭环

1. **采集**：当前随机初始化/持续更新的 actor 在真实仿真中执行带探索噪声的动作；记录物理状态、实际动作、传感器、控制器状态、随机化上下文和真实终止。回放保留失败，区分失败、超时和 reset 后首帧。
2. **拟合 F**：从本轮及历史回放取连续窗口，用记录的动作自回归预测；拟合无传感器噪声的状态目标及必要的终止/奖励项。监督中不跨 episode，不把未来观测输入自由预测。
3. **恢复起点**：从真实回放恢复物理状态和 actor 历史。增强组额外用实际历史恢复 WSM 后验；缓存注明先验版本、控制器和 episode 边界。
4. **更新 actor**：在 F 中展开短时轨迹，计算折扣奖励及 critic 尾值，BPTT 更新 π 和 ψ。此阶段冻结 F、Q、critic 的参数；不能用包裹整个前向过程的 `no_grad` 切断动作梯度。
5. **更新 critic**：从分离计算图的想象轨迹构建 TD(λ) 目标。保存完整优化器、随机状态及计数；新 actor 回到步骤1。

模型初始拟合和探索预热均计入成本。先以有界试跑验证上述循环，不能要求尚未学习的随机策略接近发布策略，也不能把旧冻结 RSSM 的方向检查当作新 F 永远不得训练的理由。数值、数据边界、断点完整性等工程检查继续执行；方向检查用于诊断训练中的模型偏差。

### 4. 必须补齐的先验想象接口

真实执行时，Q 使用实际传感器得到后验状态；想象中没有未来相机图像。不能复用真实未来的 h，也不能一直冻结 h_t，却声称已经实现持续变化的历史先验。

目标方案是按用户给出的递归公式，用 **SM 预测的观测或 encoder 特征**更新冻结 WSM 的完整循环状态。现有 WSM 还需要深度/触觉，因此须先实现并验证观测或特征桥接；仅预测关节状态不能直接满足接口。具体数据流见 [ARCHITECTURE.md](ARCHITECTURE.md)。

此前的 action-only WSM prior 展开保留为较简单的候选/消融，它不读取 F 的预测观测，不能声称与上述公式等价。接口验收比较同一记录动作下的特征偏差、动作偏差及自由展开/物理结果；冻结 WSM 参数但保留输入梯度。正式比较前选定接口，两条路线均不能使用真实未来观测修正 actor 想象。

原 SM 的控制频率和本项目不同。先以短窗口验证状态与动作对齐，再在两组共同的候选想象长度上做吞吐和累积误差试跑；正式比较前固定长度。不能仅因为旧 RSSM 验收了4步就永久把新的 SM 模型限制为4步，也不能直接照搬32步而忽略其物理时长。

### 5. 对照顺序与交付

| 阶段 | 具体任务 | 交付与验收 |
| --- | --- | --- |
| A：SM 基线 | 新 actor/critic、新 F；接通采集→模型拟合→BPTT→critic→再次采集 | 一次有界完整试跑；证明 actor 更新后实际用于下一轮采集，模型持续更新，断点可续；保存真实学习曲线。并与同任务随机初始化的 PPO-Sim 做方法/成本对照 |
| B：先验接入 | 在 A 上加入冻结 Q 和可训练 ψ，完成真实后验与想象 prior 的状态接口 | 初始公共 actor 参数匹配，额外分支初始不改变输出；记录先验来源、输入、冻结检查、梯度及部署一致性结果 |
| C：主比较 | SM 与 SM+prior，各两个配对训练种子；同任务、奖励、控制器、随机化、评估和资源上限 | 精度—成本曲线、逐种子结果、最佳 checkpoint、逐回合指标及失败案例；完整核算先验额外开销 |

PPO-Sim 用于阶段A确认学习能力和成本参照；主实验的因变量是**加入先验后精度是否改善**，不能用“两卡 PPO、两卡仅训练 RSSM”代替阶段C。两种子仅支持探索性结论。

SM+prior 比基础 SM 增加了表征分支和视觉信息。即使主比较成功，也只能先归因于这套增强方案；若要单独证明“预训练知识”有用，再增加同传感器/同容量的从头表征或打乱先验对照。不把更多输入、更大网络的收益直接等同于知识迁移收益。

后续机制实验按顺序安排，不在第一轮同时叠加：

| 要区分的因素 | 对照 | 能支持的结论 |
| --- | --- | --- |
| 额外容量/传感器与预训练知识 | 相同可用模态、匹配容量的历史编码器/从头 WSM vs 预训练冻结 WSM；随机冻结 WSM 仅作负对照 | 是否存在超出输入或参数增加的预训练收益；短时打乱先验只作依赖诊断 |
| 上下文一致性 H2 | 同接口、同数据与辅助预算，有/无多步上下文监督 | 状态/动作偏差减少是否对应真实控制改善；检查特征塌缩 |
| 动力学辅助 H3 | 仅 actor 读取 h vs actor 与 F 都读取 h | 同起点、同记录动作的物理多步预测及动作响应是否改善；保留原基线的物理状态与 ξ |
| 先验适配 | 冻结 vs 仅辅助目标微调 WSM | 收益能否抵消额外成本；更新后重算状态，单独检验部署一致性 |

### 6. 精度与训练成本如何衡量

当前连续旋转任务没有唯一正确的关节动作；不使用“与发布 actor 的动作 MSE”衡量精度，也不只看总回报。

| 指标 | 含义及约束 |
| --- | --- |
| 旋转轴偏差（角度均方根/高分位） | 物体初始局部目标轴旋转后与目标世界轴的夹角；越小越好 |
| 非目标轴运动 | 每步旋转分量在目标轴正交方向的幅度；与目标轴累计进度一起报告，避免静止策略获胜 |
| 有符号旋转量、掉落/倾斜/停转率、回合长度 | 检查精度改善是否以少转、提前终止或只保留成功回合为代价 |
| 回报与动作变化率 | 保留原任务收益及动作平滑度作辅助指标；平滑度不能代替任务精度 |
| 同等能力的成本与同等成本的能力 | 回报/旋转/失败约束满足时达到预定精度所需 wall time、GPU-hours、真实交互量；另报固定预算末值 |

正式训练前，根据独立的基线测量固定数值目标、评估频率和停止规则；不根据两组比较结果调目标。沿用固定开发/测试划分思想，重采包含所需物理量的评估记录；旧回报日志不足以追溯全部精度指标。按回合先汇总、再按训练种子分别报告，并记录失败，避免幸存者偏差。若两组未达到同一目标，报告未达到及预算，不能编造加速倍数。

成本同时报告完整在线循环耗时及包含准备/评估的总成本。WM-Craft 已有先验作为外部资产注明，未知的原始预训练成本不能记为0；本地推断、缓存、桥接及适配成本全部计入增强组。

### 7. 本次代码核查结论与运行边界

**当前代码没有完成上述 SM 基线复现。** 现有33项测试证明的是旧适配路径的部分数值、边界和恢复正确性，不能当作新闭环验收。

| 已有实现 | 与新实验的差距 |
| --- | --- |
| `unionmimic/sm.py`：加载发布 actor，在 WM-Craft RSSM 中 BPTT，附加原 actor 动作约束 | 需独立初始化 actor/critic、独立 SM 物理动力学模型，去掉发布 actor 约束；不能只改 actor 输入维度 |
| `unionmimic/collect.py`：观测/执行动作/后验采集及终止对齐 | 需独立 actor 接口，以及 F 所需的物理状态、上下文和精度评估字段；旧数据不含完整物理监督 |
| `unionmimic/comparison.py`：PPO、RSSM 拟合或旧式 SM 分阶段任务 | 当前未把在线 F 更新与 BPTT 接成同一策略闭环；需要相应联合 checkpoint 和恢复协议 |
| `unionmimic/prepare.py`、`recovery.py`：WSM 起点恢复及来源校验 | 可复用原则，新增独立 SM 状态和先验版本；不可混用过期 posterior |
| 第三方 `sm-reference-audit` | 可比照训练步骤、TD(λ)、critic 更新和 replay；不是官方 SM 实现，也不是现有22自由度灵巧手环境的直接替代 |

首次长训检验的是已训练策略在冻结 RSSM 上的微调；当前 `adaptation-20261102-v1` 检验 PPO 微调和 RSSM 适配，二者都不能支持“SM 从头学习失败”的结论。旧动作方向检查只描述当时模型及采样状态附近的局部可靠性。

旧适配任务已按用户新指示停止，保留停止状态与全部已落盘证据；不自动续跑。下一轮资源安排扣除已用预算，遵循上面的任务顺序，不能把未花费的旧预留当成额外新额度。旧两小时巡检保持关闭；用户已授权本轮W&B登录和同步，现已验证。

---

## 历史方案：发布策略微调（已由上述主实验替代）

**Measure whether SM fine-tuning improves task quality or total adaptation cost under matched conditions.** Comparative results are pending.

## First comparison

Fix the hand, single object, released checkpoint, z-axis objective, sensor inputs, reward, action mapping, and perturbation distribution. If the original task is saturated, choose a harder setting before comparing methods.

| Run | Role |
| --- | --- |
| Released policy, frozen | Starting capability reference |
| PPO fine-tuning | Original optimization route and training-cost baseline |
| RSSM + SM fine-tuning | Short imagined-rollout gradients under test |

Keep trainable modules and model-update budgets explicit. Count SM's extra calibration. Report total incremental cost and any shared preparation cost. Frozen inference is not a training-speed baseline.

Debug with one seed, then use two paired training seeds and 100–200 held-out evaluation episodes per trained policy. Match evaluation seeds and resets. Two seeds support an exploratory comparison, not a strong significance claim. Show per-seed results; do not treat evaluation episodes as independent training seeds.

## Metrics

| Question | Record |
| --- | --- |
| Does control improve? | Rotation progress, drop rate, episode length and original task return |
| Is adaptation cheaper? | New transitions, wall-clock time and GPU-hours to a predefined quality threshold |
| What happens at fixed budget? | Quality at matched compute and separately at matched interactions |
| Where does time go? | Collection, calibration, model fitting, actor/critic updates, evaluation, startup |
| Is prediction usable? | One-/four-step prior error, reward error, failure recall and false-positive rate |
| Is the actor exploiting model error? | Imagined-return improvement versus simulator-return change |

Select numeric quality thresholds before comparing methods; the runtime check does not define them. Keep raw and scaled rewards separate. Record GPU, environment count, graphics mapping, precision and warm-up policy. Do not extrapolate GPU times using peak FLOPS.

## Before the first actor update

1. Check time alignment, terminal observations and episode-level splits.
2. Restore RSSM and controller states from real history.
3. Compare reward/failure predictions against constant or persistence references on held-out episodes, including failures.
4. Match adapted deterministic actions to the original player on real observations.
5. Check finite actor gradients and changed actor parameters; model and critic stay fixed during the actor step.
6. Save, reload and evaluate in the original simulator.

These diagnose implementation errors; passing them does not demonstrate an efficiency gain.

## Follow-up experiments

| Hypothesis | Change one factor | Decision evidence |
| --- | --- | --- |
| A short prior can train the actor | PPO vs SM with matched initialization | Physical quality and total cost |
| Longer imagination helps while predictions remain useful | H = 4, then H = 8 | Contact-conditioned error and simulator performance |
| Touch improves prediction around contact changes | Matched model with/without contact input | Prediction error and physical drop/recovery behavior |
| Dynamics reuse saves adaptation cost | Second object: frozen, fine-tuned, from scratch | Additional data and compute at matched quality |

Run these after the integrated comparison. Continuous tactile fields, other embodiments and hardware deployment remain later work.

## Report all outcomes

Keep failed runs, divergence and threshold misses. Separate alignment errors, model errors, gradient instability and runtime cost. A working integration and an unsuccessful efficiency hypothesis are distinct outcomes.

Each comparison should link configuration, code revision, checkpoint hash, seeds, per-seed metrics, cost breakdown and failure examples. Videos need playback speed, policy identity, task conditions and failure frequency. Until then, the homepage shows runtime evidence and this protocol only.

## First-run follow-up protocol — 2026-11-01

Historical status at this protocol's creation: **pretraining review package prepared; comparison training had not started**. Pilots and a separate PPO/RSSM run subsequently launched; see `TRAINING.md` for actual execution. The original four-card run has ended within its budget, and its two-hour watchdog is disabled. The budgets below describe the then-proposed experiment, not an extension of that run.

Final first-run evidence is in `work/analysis/first-run-final/`: `final-summary.json`, `data-audit.json`, reward decomposition probes, gradient decomposition probes, and `report.html`. The earlier report under the training run is an interim snapshot and must not be used as the final total.

### 1. Establish a measurement that can distinguish improvement from noise

Deliverable: a fixed evaluation manifest, episode-level return/reward-component/rotation/drop records, and paired uncertainty estimates. Keep seed 123 as development history only. Use 16 development seeds × 8 episodes for final checkpoint selection and a disjoint final set of 25 seeds × 8 episodes. Periodic checks may use a fixed 32-episode development subset; they do not replace the full selection/test evaluations. Store simulator parameters and reset states, not only seeds. First verify replay fidelity; GPU PhysX is not guaranteed to reproduce a trajectory from the seed alone. If reset replay is imperfect, report batch-level variability and do not claim exact episode pairing.

Repeat the initial policy on at least four development batches, then evaluate the selected checkpoint on the same conditions. Select checkpoints and failure thresholds on development data only; open the final set once after selection. Bootstrap by reset batch within each training seed, and report the two training seeds separately. Evaluation episodes are not additional training seeds.

Proposed practical target, to freeze before comparison: mean original return at least 5% above the independently measured starting policy, no lower mean signed rotation, and no material increase in failure/drop rate (predeclare a 2 percentage-point noninferiority margin; report the confidence interval, including when the sample is insufficient). This is an engineering target, not a predicted result. Do not lower the existing prediction or recovery gates.

### 2. Test whether model gradients predict a useful action change

Deliverable: `action-sensitivity` results for 64 held-out starts spanning stable rotation, contact changes, incipient slip, and early reset failures. Save the full simulator/controller/DR/RNG state and replay history to restore contact state. Validate two unperturbed continuations before interpreting any perturbation; if identical replay fails, estimate repeated-rollout noise and require effects above that noise floor.

For each start, compare ± perturbations in eight normalized action directions at amplitudes 0.01, 0.03 and 0.05, with identical exogenous noise where possible. Measure H=1 and H=4 real and imagined changes in return, signed rotation, joint position, and failure probability. Report rank correlation, sign agreement, gradient projection versus finite differences, and confidence intervals by contact regime. The pass condition for continuing BPTT investigation is positive correlation and sign agreement demonstrably above chance on nonsaturated actions; forward RMSE alone is insufficient. Large perturbations and contact discontinuities require ranking tests rather than assuming a smooth true derivative.

If both prediction ranking and gradient direction fail, prioritize action coverage/dynamics supervision. If ranking is useful but BPTT direction fails, investigate the categorical straight-through estimator and optimizer/critic; compare PPO inside the same WM. Do not immediately increase imagination length.

### 3. Separate data, critic and model failures with bounded ablations

Run these sequentially, not as a full grid. Each optimization ablation gets the same checkpoint, replay split, random starts, 500 actor updates and three update seeds, followed by the development evaluation. Match the observed action-change budget so a smaller update is not mistaken for a better method.

| Test | Change | Decision evidence |
| --- | --- | --- |
| Data coverage | Fixed-size replay: current rolling sampling vs reservoir sampling stratified by contact/failure/action deviation | Held-out action sensitivity and physical return at equal unique transition count |
| Data quantity | Nested 25%, 50%, 100% datasets with the same strata and head-fitting budget | A learning curve that isolates quantity from coverage; more routine samples alone need not help |
| Reward information | Audit prediction errors separately for signed rotation, hand pose, control error and drift; fit diagnostic component heads on fixed frozen features | Whether low total-reward error hides a poor prediction of the component the actor must improve; diagnostic heads do not change the task reward |
| Critic | Tail coefficient 1 vs 0.25; then a critic initialized with real discounted-return targets | Physical improvement and reward/tail gradient alignment; zero-tail is a diagnostic only, not automatically a replacement objective |
| Anchor | Coefficient 10 vs 3, holding actual action drift comparable | Whether the original-policy constraint prevents improvement or protects against model error |
| Model | Frozen RSSM vs a small dynamics/residual-model update with held-out multi-step state/action supervision | Better counterfactual ranking without loss of deployed-policy behavior; invalidate/rebuild latent caches after any encoder/RSSM update |
| Horizon | H=4 vs 8, only after direction tests pass; consider 16 later | Gains in real return and failure behavior, not imagined return alone; horizons are 0.4/0.8/1.6 seconds here |

Keep early failure episodes in their own evaluation stratum. The current warm-up of eight steps excludes failures before usable starts, so padding/replaying reset history needs a separate correctness check. Build a locked failure test set with at least 100 distinct failure episodes; tune the probability threshold elsewhere. Distinguish naturally occurring failure prevalence from stress-test prevalence when reporting precision/calibration.

### 4. Compare three routes to locate the source of any gap

| Arm | Policy input | Actor signal | Purpose |
| --- | --- | --- | --- |
| A: WM-Craft + PPO in Isaac Gym | Same frozen WM feature + proprioceptive history | Actual simulator reward and PPO advantages | Required physical-learning baseline |
| B: WM-Craft + SM | Same actor input and initialization | Four-step differentiable WM rollout + terminal critic | Required comparison against A |
| C: PPO in the same learned WM | Same latent transition, heads, starts, action noise and terminal convention as B | PPO likelihood-ratio update, no dynamics derivative | Bounded diagnostic for model quality versus BPTT; not a substitute for A |

If A improves while B and C do not, suspect model rollout/coverage. If A and C improve but B does not, suspect derivative/critic/constraint design. If all fail, inspect task headroom, optimization budget and reward tradeoffs before blaming the model. If B succeeds only after dynamics fitting, the frozen RSSM was the limiting configuration; this does not establish that SM is generally inferior or superior.

The primary optimizer comparison uses a fresh critic of the same architecture and information access in A/B, with actor mean weights and input normalization copied from `work/runs/step2-20261101/calibrated/best.pt`. The upstream PPO default has an asymmetric privileged-state critic; report that as a separate practical baseline if used. Record PPO's policy distribution and exploration separately from the shared actuator noise. Keep deployment observations identical. Freeze encoder/RSSM and normalization in the primary comparison; head-fitting costs belong to B. Compare both native regularization and an explicitly matched action-drift control if the primary results differ.

**PPO warm-start must be implemented and checked.** The calibrated checkpoint contains actor and WM weights, but no PPO optimizer, asymmetric critic, epoch or frame state. Upstream full-state restore expects those keys. Load weights explicitly and initialize optimizer/critic/counters; do not rename this file and call ordinary full resume. Verify deterministic-action parity, a finite PPO update, unchanged frozen modules, timeout bootstrapping and checkpoint reload before timing a pilot.

Freeze the initial task distribution: same object/z-axis, reward coefficients, control period 0.10002 s, target integration, observation/action noise, reset/physics randomization, gravity cone 1 degree, and action-conservative scale **1.0**. The upstream defaults in Python are not the resolved YAML values. Prevent gravity curriculum advancement in both arms; restarting an environment each SM cycle otherwise differs from a persistent PPO environment. An expanded curriculum can be a second, separately defined task.

### 5. Budget and stopping decisions

The user must review the prepared launch before the **first optimizer step**, including supervised model fitting. All preflights in `work/analysis/pretraining-review-20261101/` have zero optimizer steps. The former four-card run and its watchdog remain stopped.

Configuration provenance is explicit:

| Setting | Basis and current choice |
| --- | --- |
| PPO loss | Pinned WM-Craftnet `rl_games` implementation, reused directly |
| PPO horizon / minibatch / mini-epochs | Paper Table 7: 16 / 4096 / 4 |
| PPO batch | Accumulate at least 16,384 valid samples before updating; each 16-step block retains its own GAE bootstrap |
| PPO discount / GAE / initial LR / clipping | Paper and repository: 0.99 / 0.95 / 1e-4 / 0.2; upstream adaptive KL schedule |
| V100 environment count | Measured 256; roughly 380–391 physical transitions/s including camera and RSSM, without optimizer updates |
| Critic | Project adaptation: SM's latent-state/history/controller critic, fresh optimizer, LR 3e-4; differs from upstream privileged asymmetric critic |
| RSSM fitting | Reuse native RSSM, reward/prop decoders and KL; add real-trajectory H=4 supervision and feature-drift penalty |
| RSSM pilot LR / batch / chunk | New conservative adaptation choices: 1e-5 / 8 / 32; original WM LR is 1e-4 and batch is 16 |
| Precision | FP32, PyTorch 2.1 / CUDA 11.8; V100 sm70 verified; BF16 disabled |

The next authorized-for-review launch contains **two bounded pilots**, not an automatic long comparison:

1. Physical PPO: at most 16 epochs (at least 262,144 valid rollout samples if all finish), with a 30-minute safety cap. Save every four epochs. Expected collection alone is about 11–12 minutes; actual updating/evaluation adds cost. This estimate is not time to convergence.
2. RSSM adaptation: at most 250 supervised steps, batch 8, chunk 32, four-step prior loss; 40-minute safety cap. Current forward/backward plus full-prefix reconstruction costs about 5.5–6.1 seconds per step, suggesting roughly 23–26 minutes before checkpoint/validation overhead. Fit only the fixed training split. Save every 50 updates and do not automatically start SM.

After fitting RSSM, rebuild posterior states and prior features with `prepare --recompute-state`, retaining the frozen encoder's embeddings. Old categorical samples are never forced into a changed RSSM. Any encoder change requires raw-observation re-encoding and is rejected by this path. Validate H=1/H=4 under the original gates and physically reevaluate the **unchanged actor**: its inputs changed when RSSM changed. The new checkpoint starts with `calibration_report.passed=false`, preventing an implicit SM launch. Formal adoption and online alternation require this validation; having a nonzero RSSM gradient is not evidence that fitting improves control.

A formal A/B comparison remains conditional on the pilots. Use the same actor/model starting checkpoint and two paired training seeds, 51001/51002. First isolate PPO versus SM with a frozen common representation; compare learned RSSM against frozen RSSM as a separate factor. Count additional model fitting and calibration in total cost. PPO's Gaussian action sampling differs from SM's deterministic policy plus actuator noise and must be recorded.

The researcher should set update/sample budgets, evaluation cadence, and stopping rules from source configuration and local measurements. The user supplies a resource ceiling, not a convergence forecast. The original suggested 48 GPU-hour ceiling is a ceiling for a separately reviewed experiment, not permission to restart or extend the ended run. The current manifest does **not** authorize that run.

Predeclared formal decision rules:

- Quality: at least +5% original return, nondecreasing signed rotation, and failure noninferiority within 2 percentage points; require adequate uncertainty estimates and three consecutive development checks before the locked final test. These are engineering targets, not predicted gains.
- Plateau: after at least one million physical transitions and six development checks, consider stopping if the best mean improves less than 1% across six checks. Apply this only if measured evaluation noise permits resolving 1%; otherwise improve measurement precision. A plateau means no observed progress, not proof of global convergence.
- Failure/resource: stop on nonfinite metrics, stale caches, failed model validation, repeated >5% physical-return regression, or the reviewed resource cap. Never lower acceptance thresholds after seeing results.
- Evaluate initially every 131,072 physical transitions in the formal proposal, with wall time and imagined updates logged separately; determine the final sample limit and evaluation reservation from the pilots before launching a formal comparison.

Report curves against total GPU-hours and cumulative physical interactions. Include preparation, fitting, calibration, evaluation, failures and restarts. Only compute a speedup if both arms reach the same quality target. Original WM-Craftnet's 3000-epoch downstream example is about 49.2 million transitions under 1024 × 16 collection; it does not imply a V100 GPU-hour budget. No verifiable GPU-hour figure was found in its paper or official repository documentation.

### 6. Review package and current acceptance state

`work/analysis/pretraining-review-20261101/launch-review.json` fixes inputs, source hashes, seeds and pilot limits. `approval.template.json` is explicitly unapproved. `scripts/train_reviewed_pilot.py` verifies the reviewed hashes and approval before allocating a simulator, and checks approval again before every optimizer step. No approval has been written and no pilot training has run.

Verified without updates:

- Released/calibrated actor weight-only PPO load has zero deterministic-action difference. Both actor normalization buffers remain fixed. Fresh critic and optimizer are explicit.
- Physical sampling and PPO backprop work at 16/32/64/128/256 environments. A 34-block test crosses real resets and excludes 68 incoming-reset transitions; timeout/failure GAE semantics also have CPU tests.
- The full four-mini-epoch PPO loop and checkpoint save/reload run with both optimizers disabled. Corrected local KL measurement reports zero for identical policies; the pinned upstream epsilon formula reported a negative value. PPO actor loss remains upstream code.
- Native RSSM, reward, prop and failure heads receive finite nonzero gradients; encoder and image decoder remain frozen. Full-prefix recomputation and original H1/H4 prediction gates pass using unchanged weights. Acceptance after actual model fitting remains unmeasured.
- New-cache provenance rejects stale RSSM and changed-encoder inputs unless the appropriate rebuild is requested; review guards reject unapproved or modified launches.

Action-response measurements remain **inconclusive**, and therefore do not authorize long SM/BPTT training. Tensor restoration alone did not reproduce contact outcomes. Fresh-process action-history replay reproduces exposed root/DOF states but still has physical noise and varying sensor/RSSM posterior samples. Offline comparison consequently fixes one recorded posterior for every model branch and measures physical null noise. The four-start/two-direction pilot is insufficient for the full action-sensitivity gate; all raw results are retained.

The current runnable preparation covers PPO and RSSM pilots, fixed evaluation manifests, read-only action probes and cache invalidation. Critic real-return prewarming, the broader stratified action test, PPO-inside-WM, and a full online RSSM/SM alternation are subsequent gated experiments, not claimed as completed implementations. No long training should be launched directly from these preflights.

### 7. What is reproduced and what is adapted

WM-Craftnet provides the physical task, sensors, actor/RSSM architecture, pretrained weights, and PPO loss. Our prior `unionmimic/sm.py` implements four-step actor BPTT, a latent critic, failure modeling and replay integration; it is a project-specific adaptation of SuperMimic ideas. The pinned `sm_reference` repository is third-party, not an official SuperMimic release. This is not a complete reproduction of either paper's training setup.

Original WM-Craftnet updates its WSM separately from PPO using online replay. SuperMimic alternates real collection, multi-step model fitting, and actor BPTT while freezing model **parameters only during the actor update**. Our first run instead froze RSSM for the whole run and fitted only prediction heads. That choice stabilized initial state/action alignment but left model adaptation untested. The new fitting pilot addresses that gap incrementally; it does not yet reproduce SuperMimic's deterministic physical-state/context model or all full-WSM auxiliary heads.

Primary references: [WM-Craftnet paper, Table 7 and Appendix A](https://arxiv.org/html/2609.07002v1), [official pinned repository](https://github.com/sharpa-robotics/WM-Craftnet/tree/a1e825acaf50ade298e30b0b13c6671627af3a1e), [SuperMimic paper, Section 3 and Algorithm A.1](https://arxiv.org/pdf/2610.28480).
