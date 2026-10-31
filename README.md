# unionmimic

在 WM-Craftnet 中保留视触觉 RSSM 状态估计，接入 SuperMimic 式短段预测、策略反传和价值训练。首轮只做 Sharpa 单物体 z 轴旋转，复用发布权重，与相同起点的 PPO 微调比较。

步骤 1 已完成：单张 V100 上通过 CUDA、GPU 物理/深度相机及原策略 200 步回放，三个进程均正常退出。见 [验收记录](docs/STEP1.md)。第 2、3 步已细化适配方案，尚未实施或启动训练。

## 源码和运行环境

- 上游版本固定在 `config/upstreams.json`，完整源码存放在 `work/upstreams/WM-Craftnet`。
- Python 3.8、PyTorch 2.1/CUDA 11.8、Isaac Gym Preview 4，使用项目独立环境。
- 环境、安装包、权重、数据和日志存放在 `.venv/`、`work/`，不提交 Git。
- `source scripts/env.sh` 设置项目路径及缓存。服务器专属代理配置放在忽略的 `config/network.local.env`，格式见 `config/network.example.env`。
- Isaac Gym 的凸分解缓存使用其默认目录 `~/.isaacgym/vhacd`。
- 不修改系统 Python、驱动或其他项目。

## 步骤 1 验收

```bash
bash scripts/bootstrap.sh
bash scripts/run_step1.sh --gpu 0 --steps 200
```

结果和完整日志保存于 `work/results/step1/`。可使用 `--stage cuda|gym|policy` 独立复测失败环节。显式指定 GPU 序号，不重映射 `CUDA_VISIBLE_DEVICES`，使 CUDA 与 Vulkan 使用同一设备编号。

1. V100 上的 FP32 张量运算与反向传播。
2. Isaac Gym GPU 物理步进和 GPU 深度图读取。
3. Sharpa 触觉有有效值及接触响应。
4. z 轴 checkpoint 严格兼容加载，RSSM 和原策略产生有限输出。
5. 单环境运行至少 200 个控制步，记录奖励、终止、重置、显存和耗时；固定种子与物体。
6. 策略和世界模型参数保持不变，不发生 PPO 或世界模型训练。

验收快照保存在 [step1-summary.json](docs/evidence/step1-summary.json)，包含退出码、checkpoint 校验值、源码版本和验收脚本校验值。200 步用于确认运行链路；完整 episode 的成功率评估安排在后续训练比较中。

## 接续

[RSSM 与 SM 的具体适配顺序](docs/ADAPTATION.md) 包含可复用部分、必改接口、交付物和验收条件。

本阶段不做跨本体迁移、共享潜空间、真机部署或其他任务扩展。
