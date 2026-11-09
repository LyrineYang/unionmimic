# UnionMimic 本地接手与下载

只交接UnionMimic；服务器原项目工作文件保留，使用独立副本提交这次更新。代码/数据快照来源是 `f97878addf0d02e4eb810ed2f0da9a9636b0331b`；本提交新增最新评估、精选权重、交接文档与索引。

## 立即用于展示

- [下载展示ZIP（约15 MiB）](https://github.com/LyrineYang/unionmimic/raw/refs/heads/main/docs/handoff-20261109/presentation.zip)：196个素材/文档/脚本文件，先打开 `DEMO_README.md`，可直接播放52秒中文成片。无需GPU。[ZIP校验值](presentation.sha256)。
- [12项精选入口](showcase.csv)：历史SM实控、几何显示、参考动作、图表与结果，各自保留实验来源和结论边界。
- [成果索引](artifacts.csv)：9,738条，可按用途/运行ID/状态筛选；[实验索引](experiments.csv)覆盖158个运行/分析目录。
- [最新SM/PPO结果](latest-results.csv)：两个已结束运行的512首回合开发评估。两者都学会旋转；不凭不同预算的最终结果宣称SM加速比。

几何估计只用于显示，尚无策略控制收益；Sharpa转笔参考视频为运动学播放，完整50Hz接触物理验收仍未通过。旧学习曲线有数据截止，最新最终结果以本次JSON为准。

```bash
git clone https://github.com/LyrineYang/unionmimic.git
cd unionmimic
python3 scripts/verify_published_artifacts.py
python3 scripts/verify_handoff_release.py
# 单独下载展示ZIP时，在ZIP和presentation.sha256同目录执行：
sha256sum -c presentation.sha256  # macOS: shasum -a 256 -c presentation.sha256
```

## 完整数据与服务器分卷

本次优先推送GitHub。完整备份目标为36,658个普通文件，源文件约135.68 GiB；42个分卷仍在生成/核验。当前36/42卷已读回核验，已完成卷合计91.90 GiB。下面的哈希文件仅列已完成卷，最终全量清单须等待服务器verification.json与SHA256SUMS生成。

原始HDF5、回放、中间/失败实验、全部保留检查点、上游源码、Git bundle、配置和外部成果都经过文件系统盘点；没有用Git仓库代替全量备份。[归档统计](archive-summary.json) · [各卷SHA-256](archives.SHA256SUMS)。

分卷只保存在服务器，**未上传GitHub、未替用户下载到本地，不标为完整迁移**：

`/home/junqiy/projects/unionmimic-handoff-20261109/`

完整清单、排除原因、原链接与实际目标、续传/解包/本地恢复工具均在该目录。用已有服务器SSH地址替换 `SERVER`：

```bash
rsync -a --partial --progress --exclude=/publication/ --exclude=/remote-checkout/ --exclude=/verification-samples/ \
  junqiy@SERVER:/home/junqiy/projects/unionmimic-handoff-20261109/ ./unionmimic-handoff/
cd unionmimic-handoff
sha256sum -c SHA256SUMS
python3 verify_handoff.py --archives
mkdir -p ../unionmimic-restored
for f in archives/unionmimic*/*.tar.gz; do tar -xzf "$f" -C ../unionmimic-restored; done
python3 verify_handoff.py --extracted ../unionmimic-restored
```

先校验原样快照；继续开发时克隆GitHub最新版，再用 `rsync -a --ignore-existing --exclude=.git/ unionmimic-restored/unionmimic/ unionmimic/` 补入大数据，避免旧文档覆盖本次交接更新。建议为完整分卷与解包结果合计预留约300 GiB磁盘；最终体积以服务器清单为准。

## 恢复环境与继续工作

分析CSV/JSON和播放视频不需要Isaac Gym。完整GPU仿真仍要求兼容的Linux/NVIDIA环境，不能声称已在接收电脑复验。按[环境说明](../ENVIRONMENT.md)恢复隔离环境：Sharpa Python3.8.20/Torch2.1.0+cu118，LEAP Python3.11.18/Torch2.11.0+cu126。使用 `source scripts/env.sh`，先 `python scripts/check_runtime.py`。依赖锁、上游固定版本、启动脚本均保留；完整备份还含实测版本清单。

凭据、服务器代理配置和机器专用环境/编译缓存不发布；Isaac Gym SDK/安装包需按官方许可本地获取。WM-Craftnet发布先验在完整分卷中，GitHub保留下载来源。字体及29个目录外的本项目碰撞缓存已存入补充卷。稳定的开放日志前缀另存；不停止原服务、不恢复旧训练。

历史训练曾在迁移前清理旧数据/中间权重；[逐项清理引用与当前可用性](historical-removals.csv)单列这些缺失。只有清理记录而无现存文件的内容，没有被算作已恢复或已迁移。现存累计指标、最佳权重与失败记录继续保留。

先分析 `geometry-sm-baseline-20261109-v1` 与 `sharpa-paired-ppo-quick-20261109-v1` 的阶段评估/事件时间，再设计等质量时间比较。后续仍是完整Sharpa接触/参考跟踪验收，以及独立的SM与SM+先验对照；本次迁移没有新增训练或扩大旧授权。
