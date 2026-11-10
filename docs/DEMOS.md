# 可展示 Demo

**11月10日动作清单。** 目前有策略执行证据的动作是 Sharpa 角块 +z 持续旋转，SM 和 PPO 均完成了同协议开发评估。冻结几何估计演示复用该动作。笔任务有单轴／多轴运动学参考，C1 重定向保持和 C2 扰动恢复已有任务接口及物理诊断。最新笔候选的留出首秒接触检查未通过，四项笔动作均没有已验收策略。−z 角块旋转列为一小时训练的新动作候选。[策略评估与新对照](SM_PPO_VALIDATION_20261109.md) · [笔任务门槛](PEN_ADAPTATION_AND_QUEUE_20261109.md)。

**优先看新成片：** [52秒中文故事视频](../work/demos/unionmimic-geometry-story-20261109-v1/unionmimic-geometry-story.mp4)。真实SM控制、实时几何估计、独立测试结果和Sharpa参考预览，来源持续标注。

| 新素材 | 实际内容 | 限制 |
|---|---|---|
| [30秒实时几何Demo](../work/demos/geometry-live-20261109-v1/geometry-demo.mp4) | 历史SM控制，冻结WM编码器＋小读出估计当前物体中心；含动作噪声脉冲 | 几何未反馈到策略，无控制收益结论 |
| [Sharpa单轴参考](../work/demos/sharpa-reference-preview-20261109-v2/02_continuous_single_axis.mp4) | Wuji动作重定向后的Sharpa网格原速播放 | 运动学，接触物理未验收 |
| [Sharpa多轴参考](../work/demos/sharpa-reference-preview-20261109-v2/03_continuous_multi_axis.mp4) | 同上，连续多轴动作 | 同上 |

新素材分别保留`record.json`／`manifest.json`、输入权重与动作数据哈希；入口为`record_geometry_demo.py`、`render_sharpa_retarget_preview.py`和`build_geometry_demo_reel.py`。详见[实测和后备层级](GEOMETRY_AND_DEMO_BACKUP_20261109.md)。以下保留旧素材。

更新：2026-11-09。主线讲述与后续镜头见 [STORY_AND_DEMO](STORY_AND_DEMO.md)。先看24秒成片：[showcase.mp4](../work/demos/unionmimic-showcase-20261109/showcase.mp4)。视频保留原速，每段标注实际来源。

新增：[15秒上下文机制视频](../work/demos/context-mechanism-20261109-v1/context-mechanism.mp4)。真实腕部深度与4步触觉读出对照，标注离线记录动作、固定状态/递推状态；不代表控制收益。

| 内容 | 文件 | 能证明什么 |
|---|---|---|
| Sharpa + SM 角块单轴旋转，30秒 | [训练策略视频](../work/demos/sharpa-leap-z-20261104-v2/demo.mp4) | 历史单任务SM策略确实学会控制；固定种子第一环境、无挑选，不代表新笔任务 |
| Wuji 单轴转笔，19.2秒 | [完整机器人回放](../work/demos/wuji-mesh-showcase-20261109-v1/02_continuous_single_axis.mp4) | 外部参考动作内容；运动学回放，无物理rollout、无学习策略 |
| Wuji 多轴操作，19.4秒 | [完整机器人回放](../work/demos/wuji-mesh-showcase-20261109-v1/03_continuous_multi_axis.mp4) | 同上；加入机器人网格、灯光、阴影与进度条，替代骨架示意 |
| Sharpa 四种初始抓握，1.5秒 | [接触诊断](../work/demos/pen-contact-verified-20261104-v1/diagnostic.mp4) | 旧首帧候选零残差保持诊断；不是完整参考跟踪或学习效果 |

可直接用于展示的技巧：更好的镜头、光照、参考轨迹与实际轨迹叠加、失败前后对比。已实施网格回放与剪辑，不变速、不生成虚构运动。后续若用发布策略、动作教师或蒸馏加速得到演示，单独标记来源、独立保存，不混入从随机策略开始的SM/PPO基线结果。

来源清单：`work/demos/unionmimic-showcase-20261109/manifest.json`、`work/demos/wuji-mesh-showcase-20261109-v1/manifest.json`；渲染入口 `scripts/render_reference_showcase.py`、合成入口 `scripts/build_demo_reel.py`。Wuji代码固定为 `2becb7380fed86171056f9bf4338b2a6a0c14b01`。

## 主线与下一步

1. 完成Sharpa整段参考状态与接触验收，检查SM/PPO共享控制、奖励、终止、观测和想象接口。
2. 验收通过后自动启动四卡：四动作SM/PPO、仅参考动作SM/PPO。记录位置误差、轴向误差、完成率、失败率和达到同一表现的训练时间。
3. 从本轮有限checkpoint自动记录新任务真实策略视频，保持固定种子与同一相机，同时显示参考和实际动作。学习收益未产生前保留参考回放标签。
4. 稳定SM baseline后，加入冻结WM-Craftnet循环策略上下文；真实采集与每一步想象都更新上下文，物理动力学F保持独立。先验证精度和稳定性，再研究预测改善与辅助适配。

11月9日增加了可用于讲述研究必要性的 [实测总览图](../work/analysis/frozen-context-overview-20261109/overview.png)：分别呈现成本、上下文递推和额外信息，保留信息探针没有收益的结果。详见 [先验审查](FROZEN_CONTEXT_AUDIT_20261109.md)。
