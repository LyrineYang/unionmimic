# 运行环境

当前分为两条隔离运行时：Sharpa/历史实验保留项目内 `.venv`（Python3.8.20、Torch2.1.0+cu118、Isaac Gym Preview4）；LEAP原生验证使用下列Python3.11环境。不升级系统Python/CUDA/驱动，不在旧venv内安装LEAP依赖。

## LEAP原生运行时（11月3日新增）

入口：`bash scripts/leap_python.sh <原库脚本与参数>`。它直接调用固定上游代码；项目只增加运行与验收层。

| 内容 | 位置/约束 |
| --- | --- |
| 固定源码 | `work/upstreams/sm-reference-audit`，提交`0ab03eb2244b4a1a020d94ef38a4823bb9e15ba0`；不修改其原锁文件 |
| 独立venv | `work/envs/leap-native-py311`；基础解释器为已有`work/lafan1/env/bin/python3.11`，仍依赖该解释器，不能单独删除基础目录 |
| 依赖overlay | `work/runtime/leap/{pyproject.toml,uv.lock}`；同一Torch2.11改为支持V100的cu126构建 |
| glibc | `work/tools/leap-sysroot`内2.28，仅由显式loader用于该进程；宿主2.27不变 |
| 工具/缓存 | `work/tools/uv-native/bin/uv`、`work/cache/leap-native`；不改用户全局Python工具 |
| 版本与验收 | `work/analysis/leap-native-20261103/{installed-packages.json,runtime.json,package-tests.log}` |

原wheel版本/哈希保留；CUDA依赖改用NVIDIA官方镜像，Triton为原PyPI文件的已校验本地wheel。完整环境差异、原生验证和训练审批见[执行记录](LEAP_NATIVE_EXECUTION_20261103.md)。短试跑仅写本地日志，尚无W&B指标桥接；历史环境的W&B登录不等于新脚本已有同步能力。

22:46实测：CUDA矩阵反传、Warp kernel、33项原库包测试通过。下文81项检查属于旧运行时，不能用于证明新LEAP仿真/训练成功。

## 入口与隔离

```bash
source scripts/env.sh
python scripts/check_runtime.py
```

`env.sh` 选择项目解释器，禁用用户 site-packages，清除 `PYTHONHOME`，将 `PYTHONPATH` 限定为本仓库与固定 WM-Craftnet 源码。缓存和临时目录保留在 `work/`。不修改其他项目的解释器或启动配置。Isaac Gym 原生库及资产放在 `work/isaacgym`，固定上游源码放在 `work/upstreams`；它们与本项目 Python 包环境分开管理。

`config/runtime-lock.json` 保存完整实测版本与本地依赖路径；`requirements-runtime.lock.txt` 固定可安装包。Isaac Gym、PyTorch3D 按 bootstrap 中的来源与校验和安装。版本清单不能取代实际的 GPU/仿真验收。

## 更新与回退

`bootstrap.sh` 遇到已有锁定环境时只校验，不就地升级。需要改依赖时另建环境：

```bash
UNIONMIMIC_VENV_DIR="$PWD/work/envs/runtime-candidate" bash scripts/bootstrap.sh
UNIONMIMIC_VENV_DIR="$PWD/work/envs/runtime-candidate" bash -c 'source scripts/env.sh; python scripts/check_runtime.py'
```

候选环境先通过依赖检查及有界仿真，再指定给新的运行。保留原 `.venv` 供回退；活动训练的环境、代码快照和 manifest 不变。容器化可后续用于分发，目前不引入第二套未经验证的驱动/图形库组合。

## W&B

用户已授权本轮登录和同步。非秘密的 entity/project 在 `config/wandb.env`；API key 仅由 `wandb login` 保存到用户认证位置，不写入仓库、训练配置或日志。`WANDB_MODE` 可显式选择 `online`、`offline`、`disabled`，默认 `offline`。训练只上传指标与非秘密配置，不自动上传权重、回放或源码。

```bash
source scripts/env.sh
WANDB_MODE=online python -m wandb login
```

登录、网络可达和目标项目写入是三项独立检查。离线日志先落盘；确认认证与连接后可将本轮授权的记录同步。旧两小时巡检保持关闭。

本轮实测：81项版本检查和 `pip check` 通过；W&B 0.22.3真实登录、写入及服务端读回通过。项目本地代理设置使用已验证的本机端口7893，保存在忽略的 `config/network.local.env`；不改全局网络配置。已建环境的验证通过，另建候选环境的完整重装流程尚未重新执行。
