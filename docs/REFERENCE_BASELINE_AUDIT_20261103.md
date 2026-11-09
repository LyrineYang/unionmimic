# 参考实现偏离审查与代码路线建议

时间：2026-11-03约21:53，Asia/Shanghai。状态：已核对下列源码和配置，未完成全库正确性审计，也未在本机复验参考库学习成绩。本次不改活动训练源码、不启动新训练。

## 结论

当前自写SM是有工程检查和部分学习结果的Sharpa算法变体，不能视为已忠实复现LEAP/Wuji。64环境、小batch、小网络不是Sharpa适配的必然要求；没有缩放等价或学习质量证据支持把这些设置从试跑沿用到正式长训。此前“各组一致、差异已记录”只能说明内部对照条件，不能证明基线足够强。

建议保留UnionMimic的环境、数据、评测、调度和历史证据，以克隆并固定版本的参考实现为新学习器基础，只改有明确契约的Sharpa适配层。停止继续扩展自写learner或混合多个参考库的默认设置。先复验参考原任务，再迁移；克隆本身不保证训练有效。

具体首选LEAP的cube-yaw发布配方作为当前持续旋转的学习器母版；其README有从头训练命令和对应权重，关节数通过`nj`参数传递。Wuji/Dex作为参考轨迹任务的独立实现参照，不能把两库默认值拼成第三套配方。Dex在多个状态/记忆字段硬编码20关节，迁到Sharpa22关节并非改一个参数。最终迁移范围需在原生运行和适配边界检查后固定，不承诺仅换URDF即可。

## 固定版本和实际默认值

- LEAP：`Msornerrrr/in-hand-rotation-mjlab@0ab03eb2244b4a1a020d94ef38a4823bb9e15ba0`，本机`work/upstreams/sm-reference-audit`；本次检查工作区干净。
- Wuji适配：`CumulusAlpha/Dex-Supermimic@2becb7380fed86171056f9bf4338b2a6a0c14b01`；本次完整克隆到`work/upstreams/dex-supermimic`并detached checkout，未安装依赖或执行训练。
- 本项目：当前筛选使用`online_baseline`＋`online_screen`，不是表中原始`online`所有行为都仍存在；活动计划60项源码哈希另行核验。

| 项目 | LEAP yaw原配方 | Wuji/Dex当前源码默认 | 本项目现有实现 |
| --- | --- | --- | --- |
| 环境数 | 4096 | 1024 | 64 |
| F拟合batch | 2048 | 2048 | 256 |
| Actor想象起点batch | 4096；指尖发布命令覆盖为2048 | 1024 | 256，与F共用一个配置字段 |
| F主干 | 1024×3，LayerNorm＋SiLU | 2048×3，LayerNorm＋SiLU | 512×3，LayerNorm＋ELU |
| Actor | 512–512–256，ELU | 512–256–128，ELU | 512–256–128，ELU |
| Critic | 256–128，LayerNorm＋SiLU | 512–512–256–128，ELU | 256–128，LayerNorm＋ELU |
| replay向量步 | yaw1000；指尖800 | 1000 | 512 |
| 初始采样 | 64向量步 | 64向量步 | 128向量步 |
| 模型单独预热 | 300轮×10次拟合＝3000次，期间继续采集 | 500轮×10次＝5000次，期间继续采集 | 原0号200次；原3号3000次并继续采集 |
| k / H | k8/H16；指尖发布命令H32 | k8/H32 | k8/H8；本轮分支H8或16 |
| UTD，轮/向量步 | 0.4 | 0.4 | 原0.125；本轮0.125或0.4 |

来源：[LEAP训练器](https://github.com/Msornerrrr/in-hand-rotation-mjlab/blob/0ab03eb2244b4a1a020d94ef38a4823bb9e15ba0/scripts/wm/train_yaw.py)、[LEAP网络](https://github.com/Msornerrrr/in-hand-rotation-mjlab/blob/0ab03eb2244b4a1a020d94ef38a4823bb9e15ba0/src/in_hand_rotation_mjlab/world_model/policy.py)、[LEAP动力学](https://github.com/Msornerrrr/in-hand-rotation-mjlab/blob/0ab03eb2244b4a1a020d94ef38a4823bb9e15ba0/src/in_hand_rotation_mjlab/world_model/model.py)、[Wuji配置](https://github.com/CumulusAlpha/Dex-Supermimic/blob/2becb7380fed86171056f9bf4338b2a6a0c14b01/src/dex_supermimic/rl/dex_supermimic/config.py)、[Wuji网络](https://github.com/CumulusAlpha/Dex-Supermimic/blob/2becb7380fed86171056f9bf4338b2a6a0c14b01/src/dex_supermimic/rl/dex_supermimic/networks.py)。

必须区分源码默认值、发布checkpoint实际配方和本机实测。Wuji的`setting_hand2_pen_spin.md`把critic写为512–256–128，而当前`networks.py:148`实际是512–512–256–128；源内文档也可能落后。上表按执行源码，不用README/说明文件代替最终解析配置。

## 比网络大小更重要的差异

| 差异 | 本次核对结果 | 判断和后续处理 |
| --- | --- | --- |
| 状态积分 | LEAP预测速度/位置修正并作物理积分与旋转指数映射；Dex使用SO(3)增量；本项目预测大状态向量加性残差，再归一化四元数 | 算法/模型设计变化，非手型尺寸适配。不能仅把512改1024视为复现 |
| F损失尺度 | LEAP以典型单步变化定标；Dex用delta统计和物理量下限；本项目损失按状态标准差缩放并按维度/时间取均值 | 小量但关键的接触/速度变化可能获得不同权重；影响是否造成失败尚待配对验证 |
| 归一化与截断 | 本项目actor输入标准化后额外clamp±10，F输入±20；参考有各自统计/epsilon/冻结规则 | 保留参考语义为默认；任何截断需列明，不按通用“稳定性”解释直接加入 |
| 失败学习 | 原0号BCE通过预测状态更新F，LEAP参考该路径detach；现候选已隔离。失败采样、校正及想象概率处理仍有实现差异 | 这是已发现且有控制改善证据的偏离，但并非唯一原因已查清 |
| Critic目标 | LEAP yaw对TD(λ)目标按任务奖励上限裁剪，本项目没有该规则；Wuji当前runner仍从想象更新critic统计 | 真实数据统计不是两库共同默认。原生复验先不改；Sharpa裁剪须按其奖励推导，不能硬搬LEAP数值 |
| 学习率/权重平均 | LEAP yaw固定LR；其指尖发布配方有衰减、EMA及失败头等设置；Dex有余弦调度；本项目固定LR、没有指尖配方EMA | 按任务选择一个完整recipe，不能把指尖论文/README成绩归给yaw默认配置 |
| 动作/时间/奖励 | Sharpa22关节、触觉历史、控制器/PhysX、约10Hz；LEAP yaw20Hz，Wuji pen50Hz；本项目奖励另乘0.01 | 属于必须明确的任务差异。H步数、γ及成本不能脱离真实秒数比较；移植后不声称原论文同协议分数 |
| PPO参照 | 本项目调用rl_games损失并自写循环；Wuji高斯策略尺度、rollout100、γ0.996等另有规定 | 本项目PPO有学习结果，但不能称Wuji原生PPO已复现；主比较要保留任务与信息公平，算法配方分别核对 |

以上是定向审查，不是所有缺陷的总数。已有数值/恢复测试证明所测契约，不证明上述设计变化有效。需要给每一条差异补“保留/必要适配/待消融/撤回”及证据，而不是继续叠加调参。

## 哪些资源改动有依据

本机实测V100 capability=7.0，当前Torch报告`bf16_supported=False`。Wuji默认BF16不能原样开启；FP32是明确硬件例外，需记录吞吐/内存影响。该事实不能用来解释此前未经容量实测就缩小环境、MLP和batch。

64环境可以用于有界工程试跑，不能自动成为正式复现规模。与LEAP相比，本项目F batch小8倍、想象起点小16倍、环境小64倍；初始128×64＝8192条原始转移，而参考64×4096＝262144条，相差32倍。保持batch/env固定比例也不保证学习等价；两个batch应分别配置。

撤回刚才四动作草案中“2048环境，拟合/想象均8192，H32，默认每卡双进程”作为长训默认配方。它是容量提案，没有原配方或学习证据。24GiB改为观测目标，先测完整参考配方单进程；无必要不通过改模型、视野或绑定batch来凑显存。

参考原生依赖需要隔离环境：LEAP Python>=3.10、Torch2.11、固定mjlab/Warp；Dex Python>=3.10及另一个mjlab版本。本项目Isaac Gym为Python3.8/Torch2.1。先核验原生栈在V100/现有驱动上的支持，不升级全局驱动或覆盖旧venv。Sharpa迁移的运行方式根据兼容性再决定；进程桥接也有吞吐/数据边界成本，不在此假定已经解决。

## 建议的迁移验收顺序

1. **固定母版和原生控制。** LEAP yaw原命令/网络/统计/更新规则，先独立环境、发布权重回放和工程检查，再报告从头学习的试跑。原生长训仍需用户正式确认；原生LEAP只作阳性控制，不替换Sharpa主线。
2. **保留learner，列清Sharpa适配。** 关节数/顺序/限位、单位与坐标、控制器、终止前状态、传感器历史、FK/接触/奖励映射各有来源和测试。核心网络、优化器、损失和回放不凭印象重写；真正必须改的地方形成小补丁和差异表。不能在没有证据时承诺所有任务字段都可直接复用。
3. **同批数据核对更新。** 使用固定权重、记录数据和RNG，对比F输出/损失、奖励与存活、TD(λ)、actor梯度及一次优化器更新；再做真实/想象控制器、观测和终止边界验证。先证明接口适配没有另造一套算法。
4. **Sharpa学习与性能。** 从原生规模/独立batch起步测容量；仅有OOM或实测瓶颈才做显式缩放。少量有界SM/PPO学习试跑后报告，再确认长训。当前分支筛选的赢家不能代替这项验收。
5. **恢复2＋2动作和先验比较。** 保留两项能力验证＋两项真实参考动作的任务方向，待可靠学习器/适配成立再固定其规模、并发和时长；最终仍是SM与SM+WSM比较。

保留现有模拟器安装、Sharpa资产、终止/动作对齐记录、RSSM证据、评测与预算管理及全部历史断点；现有自写learner保留为历史诊断分支，不再默认为参考复现基线。当前S2/S5按原已授权23:20硬截止继续，未重启、延时或取消；本次未获停止它们的明确指示。

机器记录：[源码与默认值审查](evidence/reference-baseline-audit-20261103.json)。此报告使前一版四动作长训参数处于待重审状态，不更改ROADMAP交付职责。
