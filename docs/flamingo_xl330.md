# Flamingo（单腿站）— XL330 训练方案

单腿站和双脚站可以随时切换，左右腿都能做；抬起的那条腿向后折（"小天鹅"姿态）。
任务 ID `Mjlab-Flamingo-Flat-Backlash-MicroDuck`，配置在
`src/mjlab_microduck/tasks/microduck_flamingo_env_cfg.py`，奖励和指令实现在
`src/mjlab_microduck/tasks/mdp.py` 的 flamingo 部分（7722 行起）。

**当前推荐的检查点：`2026-10-30_11-23-07_flamingo_v31_ws` @ `model_1000.pt`**，导出文件是 Mac 上的
`~/ckpts/policies_flamingo/flamingo.onnx`。

## 1. 什么是"XL330 方案"

两套方案只有被训练的那个物理模型不同，任务和奖励一样：

| | XL330 方案（本文） | HD1910 方案 |
|---|---|---|
| 分支 | `flamingo`（基于官方 `develop`） | `flamingo-hd1910`（基于 `ours-813g-hd1910`，文档 `docs/flamingo_hd1910.md`） |
| 舵机模型 | BAM M6 `xl330`，`kp_fw=200`，电压 6.5–8.2 V，延迟 3–6 步（`microduck_constants.py:163-176`） | BAM M6 HD1910，`kp_fw=5`，7.4–8.0 V |
| 质量分布 | 官方导出模型，737 g | `ours_835g`，835 g |
| 用在哪 | 官方硬件（15 颗 XL330） | 我们的 HD1910 克隆机 |

在 `flamingo` 分支上，舵机和质量都是写死的官方配置，没有环境变量可以切换。

**XL330 版的策略不要直接上 HD1910 的机器。** 同一个 model_1000 的测试结果：

- XL330 训练环境：32/32 能站住。
- HD1910 + 835 g 训练环境：13/16。
- `duck_runtime` 仿真（HD1910 + 835 g，6 个种子）：右脚撑地 1/6，左脚撑地 0/6，多数抬腿后 2–3 s 摔倒。

## 2. 任务定义

**指令**（`FlamingoCommand`，`mdp.py:7737`）占观测的 twist 位置：`twist = [flag, side, 0]`。

- `flag` 为 0 表示双脚站，为 1 表示单脚站。
- `side` 为 +1 表示右脚撑地、左腿后抬，为 −1 表示左脚撑地。
- 每种状态保持 3–6 s 后重采样，一半概率采到单脚（`ONE_FOOT_PROB = 0.5`）。
- 奖励里用到的目标在 1 s 内平滑过渡（`RAMP_S`），观测里看到的是原始 flag。
- 只有原来的支撑脚落地之后，才会换另一边。

**观测和动作**跟 sitstand / velstand 完全一样：

- 观测 61 维：角速度、重力投影、14 个关节位置、14 个关节速度、上一帧动作、twist、头部指令、补零的 body 指令。
- 动作 14 维，目标位置 = HOME + 1.0 × action，控制频率 50 Hz。

**目标姿态**是在训练模型上约束求解出来的常数（cfg 第 100-145 行）：

- `POSE_RIGHT` 是右脚撑地的姿态，`POSE_LEFT` 是它的镜像。
- 抬起的脚在支撑脚后方 73 mm、离地 119 mm，躯干侧倾 24.6°。
- 每个关节离硬限位至少 0.08 rad；头部偏离 HOME 的幅度有上限（`HEAD_CAPS`）。

**奖励**（权重都在 cfg 第 198-270 行）：

| 项 | 权重 | 作用 |
|---|---|---|
| `flamingo_composite` | 4.0 | 乘积项：接触模式 × 直立 × 姿态 × (质心 + (1−α))，几项都做对才拿高分 |
| `flamingo_contact` | 2.0 | 支撑脚着地，另一只脚离地；双脚指令时两脚都着地 |
| `flamingo_swing_lift` | 1.5 | 抬起的脚高度，目标 0.08 m；要求支撑脚着地、抬起的脚离地、躯干直立才给分 |
| `flamingo_swing_behind` | 1.5 | 抬起的脚在支撑脚后方，目标 0.05 m；门控条件同上 |
| `flamingo_com` | 2.0 | 整机质心（`subtree_com`）落在支撑脚上方，std 15 mm |
| `flamingo_upright` | 2.0 | 重力投影接近指令要求的侧倾，std 0.25 |
| `flamingo_pose` | 1.0 | 腿部关节接近目标姿态，std 0.5 rad |
| `flamingo_height` | 1.0 | 躯干高度：双脚 116 mm，单脚 122 mm |
| `flamingo_stillness` | 1.0 | 姿态保持住时静止；只有接触模式正确时才给分 |
| `head_pose_tracking` | 0.3 | 头部指令跟踪，权重故意调低：头占整机质量 38 %，可以当配重用 |
| `non_sole_contact` | −2.0 | 脚底以外的部位碰地 |
| `joint_limits` | −5.0 | 所有舵机关节进入离硬限位 0.05 rad 的区域 |
| `action_over_limit` | −0.5 | 目标位置超过限位 0.3 rad 以上 |
| `servo_stall` | −0.05 | 力矩 > 0.4 N·m 且速度 < 0.5 rad/s（堵转） |
| `head_range` | −5.0 | 头部超出 `HEAD_CAPS` |

另外从 sitstand 继承了一组平滑和安全相关的项：action_rate、torque_rate、gentle_motion、body_ang_vel、angular_momentum、self_collisions、dof_pos_limits。

**终止条件**：倾斜超过 50°（`fell_over`）、超时（单回合 12 s）、跑出地形边界、NaN。

**课程**：

| 课程 | 安排 |
|---|---|
| 单脚出生 | 一开始 50 % 的回合直接从单脚姿态出生；第 1500 次迭代降到 30 %，把更多数据留给抬腿过程 |
| 推力（sitstand 的一半） | 迭代 0：无；1000：±0.025；1500：±0.05；2000：±0.1；2500：±0.15 m/s |

域随机化全部从 sitstand 继承：脚底摩擦、编码器偏置、躯干和头部质心、质量和惯量、电枢、关节摩擦。执行器延迟和电压随机化在舵机模型里。

**PPO**：

- actor 和 critic 都是 512-256-128 的 ELU 网络，带观测归一化。
- lr 1e-3（adaptive，KL 0.01），熵系数 0.01，γ 0.99，λ 0.95，5 个 epoch × 4 个 mini-batch。
- 每个环境每次迭代 24 步，每 250 次迭代存一次检查点，`max_iterations` 6000，不用对称性增强。

## 3. 版本演进（每个 run 目录里的 `git/mdown.diff` 记录了训练时的提交）

| 版本 | 提交 | run | 起点 | 结果和问题 |
|---|---|---|---|---|
| v1 | `a99856c` | 未保留 | 从零 | 能单脚站，但腿是**向前**抬的：只靠腿部姿态那个高斯项，定不住抬腿方向 |
| v2 | `020245b` | `2026-10-29_17-40-04_flamingo_v2_swan`（训到 5999） | 从零 | 加了 swing_behind，变成后抬。站住、抬腿、放下都是 32/32，但有 5 个关节顶死在硬限位：head_roll 指令 +7.05 rad，限位只有 0.44；头扭了 116°；推力加到 ±0.3 后训练曲线开始下降 |
| v3 | `5fb0be3` | `2026-10-30_10-06-04_flamingo_v3`（看到 model_1000 就停了） | 从零 | 加了防顶死的几项代价、头部范围限制，推力减半。新问题是"撑脚架"作弊：抬起的脚尖戳在地上，两只脚其实一直都着地，支撑腿还蹲下去（膝 −1.32 rad，躯干 94 mm） |
| v3.1 | `1feb0e9` | `2026-10-30_11-23-07_flamingo_v31_ws`（训到 3000） | **从 v2 的 model_1500 热启动** | 抬腿两项奖励改成要求抬起的脚真的离地，并加上躯干高度项，作弊没有了。**推荐用 model_1000** |

v3.1 的训练曲线（迭代号是热启动后重新计的数）：

| 迭代 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| composite | 2.29 | **2.59** | 2.54 | 2.40 | 2.21 | 2.08 |
| fell_over | 1.66 | **1.09** | 1.61 | 2.32 | 3.43 | 3.71 |

1000 次迭代之后，推力越来越大，摔倒也越来越多。

## 4. 训练命令（5090 工作站，仓库 `~/hl/Microduck/mdown`）

```bash
git fetch origin && git checkout flamingo && uv sync

# 冒烟测试（几次迭代能跑起来就行）
uv run train Mjlab-Flamingo-Flat-Backlash-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5

# 复现推荐的这条链路
#   第 1 步：v2，从零开始（训练时是提交 020245b；用 1feb0e9 的代码从零训练没有验证过）
git checkout 020245b
uv run train Mjlab-Flamingo-Flat-Backlash-MicroDuck --env.scene.num-envs 4096 --agent.run-name flamingo_v2_swan
#   第 2 步：v3.1，从 v2 的 model_1500 热启动
git checkout flamingo
MICRODUCK_WARM_START=1 uv run train Mjlab-Flamingo-Flat-Backlash-MicroDuck \
  --env.scene.num-envs 4096 --agent.resume True \
  --agent.load-run <v2 的 run 目录名> --agent.load-checkpoint model_1500.pt \
  --agent.max-iterations 3000 --agent.run-name flamingo_v31_ws
```

几点说明：

- `num-envs 4096` 是推荐值，run 目录里没有记录当时实际用的是多少（`params/agent.yaml` 只记 PPO 参数）。
- 热启动必须加 `MICRODUCK_WARM_START=1`（`mdp.py` 第 122-144 行，Patch 5）。不加的话，迭代计数从 1500 接着算，出生课程和推力课程会直接跳到后面的阶段。v3.1 的奖励和 v2 不一样，所以用热启动，不用同任务 resume（区别见 `docs/sitstand_policy.md`）。
- 想试"当前代码从零训练"：不切到 `020245b`，直接在 `flamingo` 分支上跑第 1 步那条 train 命令（默认 6000 次迭代）。这条路没有跑过。

## 5. 评估（`scripts/flamingo/`，在 CPU 上跑）

```bash
# 选部署用的检查点：每个检查点输出一行，默认分别测无推力和 ±0.15 推力
uv run python scripts/flamingo/compare_ckpts.py logs/rsl_rl/microduck_flamingo/<run>/model_{1000,1500,2000}.pt
# 分场景细看：站住、抬腿、放下、空闲
uv run python scripts/flamingo/eval_scenarios.py <ckpt> --scenario hold lift down idle --push 0.15
# 录视频：双脚 → 右脚 → 双脚 → 左脚 → 双脚（服务器上需要先 apt install libosmesa6）
uv run python scripts/flamingo/record_video.py <ckpt> flamingo.mp4 --num-envs 3
```

这些脚本用的都是训练环境本身，只是去掉了课程、关掉了指令重采样，出生概率和推力由参数固定。

v3.1 的结果（XL330 训练环境，无推力，32 个环境）：

| 检查点 | 成功 | 侧倾 | 抬脚高度 | 顶限位的关节数 | 最差关节力矩 p95 | 抖动 |
|---|---|---|---|---|---|---|
| **model_1000** | 32/32 | 30.0° | 138 mm | 4 | 0.28 N·m | 0.024 |
| model_1500 | 32/32 | 30.0° | 139 mm | 4 | 0.26 N·m | 0.023 |

- model_500 在 ±0.15 推力下：站住 26/32，抬腿 29/32；抬腿指令发出后 0.3 s 离地。
- model_1000 的抬腿场景（8 个环境，左右各半）：都成功，抬脚 134 mm，在支撑脚后方 66 mm，抬起的脚没有碰过地。

model_1000 还有这些问题：

- 支撑腿的 hip_yaw 和 hip_roll 顶在硬限位上，目标位置最多超出 0.33 rad。
- 实际侧倾 30°，比目标的 25° 大。
- 在 `duck_runtime` 仿真（XL330）里左右不对称：右脚撑地 6/6，左脚撑地 2/6。训练环境里两侧都没问题；这个仿真接触参数更硬，作为部署前的保守参考。

## 6. 导出和部署

```bash
# 导出：观测归一化会打包进 ONNX。不要手工从 .pt 转换
uv run scripts/export.py Mjlab-Flamingo-Flat-Backlash-MicroDuck \
  --checkpoint-file logs/rsl_rl/microduck_flamingo/2026-10-30_11-23-07_flamingo_v31_ws/model_1000.pt \
  --onnx-file flamingo.onnx --device cpu
# 核对：输入输出是 61 → 14，和 .pt 的前向结果一致（误差约 1e-6）
uv run python scripts/flamingo/check_onnx.py <同一个 model_1000.pt> flamingo.onnx
# 打包成机器端的策略集：flamingo.onnx + manifest.json + merge_into_official.py
uv run python scripts/flamingo/make_policy_set.py flamingo.onnx policies_flamingo --servos xl330 --note "v31_ws@1000"
```

run 目录里那个和目录同名的 `<run>.onnx` 是训练过程自动导出的，不知道对应哪个检查点，部署不要用它。

**机器端**（`duck_runtime`，清单格式 schema 2；每个动作单独设的 `lowpass` 是 `duck_runtime` 自己加的字段，官方 robotd 不认）：

```bash
scp -r policies_flamingo user@microduck.local:~/microduck_rl/
# 在鸭子上：复制一份官方策略集，把 flamingo 的两个动作加到复制件里，原来的 policies/ 不动
python3 policies_flamingo/merge_into_official.py        # 生成 policies_official_flamingo/
duck duck_runtime.run --calib calib.json --set policies_official_flamingo --on-fall limp
```

- 按键：**4** = `flamingo_right`（右脚撑地，左腿后抬），**5** = `flamingo_left`。
- 单脚保持 6 s，然后双脚站 1.5 s 收势，再交回 velstand。
- 动作缩放 1.0，不加低通，和训练时一样。
- 单脚动作执行期间，runtime 的摔倒预判是关掉的，第一次上机要用手扶着。

## 7. 文件都在哪

| 什么 | 在哪 |
|---|---|
| 代码 | fork `sisjkly/microduck_rl`，分支 `flamingo` |
| 训练日志和检查点 | 5090：`~/hl/Microduck/mdown/logs/rsl_rl/microduck_flamingo/<run>/` |
| 拉回来的检查点 | Mac：`~/ckpts/<run>/` |
| 部署包（model_1000） | Mac：`~/ckpts/policies_flamingo/`（`flamingo.onnx` + `manifest.json` + `merge_into_official.py`） |
| 配置测试 | `tests/test_flamingo_cfg.py`（16 项） |
