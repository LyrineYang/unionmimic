# 结果与 Demo 发布

本次Git提交包含实际视频和测量数据，不只是本机路径链接。

- [52秒中文成片](../work/demos/unionmimic-geometry-story-20261109-v1/unionmimic-geometry-story.mp4)、[30秒实时几何](../work/demos/geometry-live-20261109-v1/geometry-demo.mp4)、全部历史Demo与来源清单。
- [几何探针与后备方案](GEOMETRY_AND_DEMO_BACKUP_20261109.md)、[三阶段实验](CONTEXT_STUDY_20261109.md)、[单任务SM复现证据与PPO比较协议](SM_PPO_VALIDATION_20261109.md)，对应逐回合JSON、图表和统计。
- 历史演示SM checkpoint：`work/runs/sharpa-leap-12h-20261103-v1/ckpt/best.pt`；几何读出、AR误差模型和随机编码器控制：`work/analysis/geometry-readout-20261109-v1/*.pt`。
- 实时几何Demo的对齐转移，以及Sharpa参考预览使用的重定向轨迹；参考预览仍不是物理策略结果。
- 项目源码、训练配置、测试、上游补丁及版本信息。上游实现／资产的作者与许可见[NOTICE](../NOTICE.md)和`docs/upstream-notices/`。

共770项测量／媒体／演示产物，约145.45 MiB，逐文件字节数与SHA256记录在[发布清单](../work/publications/results-demo-20261109/manifest.json)。普通Git clone即可获得这些文件，不依赖Git LFS或本机挂载。

```bash
python3 scripts/verify_published_artifacts.py
```

观看视频、阅读JSON与核对清单不需要GPU。重新运行仿真需要按[环境文档](ENVIRONMENT.md)安装独立运行时，并按固定版本获取WM-Craftnet、LEAP参考、Wuji来源和原发布先验。原始HDF5采集／训练回放、Isaac Gym及环境安装包、原发布WM大权重、凭据和本机代理设置不进入Git；它们不是这次测量／展示产物。数据来源、配置与哈希随结果保留。

正在运行的SM/PPO输出不会随Git提交自动更新，本次发布不将未完成训练写成最终结果。后续验收应追加新提交，保留本次快照。
