# 步骤 1 验收记录

2026-10-31，在 node13 的 GPU 0（Tesla V100-SXM2-32GB）完成。运行命令：

```bash
bash scripts/run_step1.sh --gpu 0 --steps 200
```

| 检查 | 实测结果 |
|---|---|
| CUDA | FP32 运算和反向传播通过，梯度有限且非零 |
| GPU 物理与相机 | 100 步；物体落地并产生接触力；72×96 深度图有 6048 个有效像素，张量位于 GPU |
| 模型恢复 | 发布的 z 轴策略及世界模型严格加载；源码 `a1e825a` 无修改 |
| 原策略回放 | 单环境、`set_obj6_block_corner`、种子 42；200 控制步耗时 36.13 秒；奖励总和 446.49，无任务终止 |
| 视触觉 | 72×96 深度图持续变化；5 个指尖传感器句柄有效；接触信号与物理接触力阈值一致，发生 232 次状态切换 |
| 重置与训练隔离 | 显式重置后物理和 RSSM 正常推进；策略/世界模型参数及 buffers 的前后哈希一致 |
| 进程退出 | CUDA、Gym、策略三个进程退出码均为 0 |

PyTorch 分配器记录的显存峰值为 171.83 MiB，不包含 PhysX/Vulkan 的分配。回放前的进程显存单次观测为 2784 MiB。计时只包含 200 步及验收检查，不含首次资产凸分解和模型加载。

初始位置扰动范围为 `[0.03, 0.015, 0.001]`，旋转扰动配置为 `5.0`，保留原传感器噪声和动力学随机化。同种子两次回放的总奖励为 445.47 和 446.49；这次交付是运行验收，尚未测完整 500 步 episode 的成功率，也未启动 PPO、世界模型或 SM 训练。

## 环境与修复

- Python 3.8.20；PyTorch 2.1.0/CUDA 11.8；Isaac Gym Preview 4；PyTorch3D 0.7.5；驱动 530.30.02。
- 用现有 GCC 11 编译原生扩展；依赖完整性检查 `python -m pip check` 通过。
- 退出崩溃定位到 NVIDIA EGL 库卸载后的渲染线程。验收入口以 `RTLD_NODELETE` 保持 `libEGL_nvidia.so.0` 的映射，仍执行仿真器清理及正常进程退出。
- PyPI、包文件域和 PyTorch 下载域可直连。此前下载 NVIDIA 安装包和 Hugging Face 权重使用的本机反代已断开；文件已缓存并有 SHA-256 校验，当前回放不依赖网络。重新下载这两项时需使用可用代理。

## 交付物

- [机器可读验收快照](evidence/step1-summary.json)、[实际运行配置](evidence/policy-config.yaml)、[依赖版本](evidence/environment-freeze.txt)。
- 深度样本：[第 1 帧](evidence/depth-0000.png)、[第 200 帧](evidence/depth-0199.png)。
- 完整运行日志在 `work/results/step1/{cuda,gym,policy}.log`，安装和诊断日志在 `work/logs/`。
- [第 2、3 步适配方案](ADAPTATION.md)：数据对齐 → RSSM 起点 → 短段预测 → actor 适配 → SM 更新 → 在线循环。
