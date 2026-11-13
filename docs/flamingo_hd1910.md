# Flamingo（单腿站）— HD1910 训练方案

单腿站和双脚站可以随时切换，左右腿都能做；抬起的那条腿向后折（"小天鹅"姿态）。目标是我们的 HD1910 克隆机：飞特 HD1910 舵机，整机 835 g，Pi Zero 2W 上跑 `duck_runtime`。

任务 ID 是 `Mjlab-Flamingo-Flat-Backlash-MicroDuck`，和 XL330 方案相同。配置在 `src/mjlab_microduck/tasks/microduck_flamingo_env_cfg.py`，奖励和指令实现在 `src/mjlab_microduck/tasks/mdp.py` 第 7710 行起的 flamingo 部分。

## 1. 什么是"HD1910 方案"

两套方案只有被训练的物理模型不同，任务、奖励和代码都一样：

| | XL330 方案 | HD1910 方案（本文） |
|---|---|---|
| 分支 | `flamingo`（基于官方 `develop`，文档 `docs/flamingo_xl330.md`） | `flamingo-hd1910`（基于 `ours-813g-hd1910`） |
| 舵机模型 | BAM M6 `xl330`，`kp_fw=200`，电压 6.5–8.2 V | BAM M6 HD1910（`actuator/params/hd1910_m6.json`），`kp_fw=5`，电压 7.4–8.0 V，最低 7.0 V，延迟 3–6 步 |
| 质量分布 | 官方导出模型，737 g | `ours_835g`：每颗舵机 +5 g，头部 +22.8 g，整机配平到 835 g |
| 用在哪 | 官方硬件 | 我们的克隆机 |

HD1910 配置在代码里的位置：

- **舵机**：`src/mjlab_microduck/robot/microduck_constants.py:184-207`。
  - `_HD1910_ACTUATOR_KWARGS` 定义 HD1910 的执行器参数。按 kissqy 那边拟合的参数，HD1910 的有效刚度约 1.03 N·m/rad，是 XL330（`kp_fw=200`）的 2 倍左右，摩擦也大很多。
  - 第 204 行是 `SERVO_MODEL = os.environ.get("MICRODUCK_SERVO", "hd1910")`，这个分支默认用 HD1910。
- **质量**：`src/mjlab_microduck/robot/mass_profile.py:37-38`（默认 `ours_835g`，可以用 `MICRODUCK_MASS_PROFILE` 改）；`ours_835g` 本身的定义在第 57-61 行。它在 `microduck_constants.py:53` 的 `spec_fn` 里生效，训练、导出、评估用的是同一个模型。
- 想在这个分支上切回官方配置做对比：设 `MICRODUCK_SERVO=xl330 MICRODUCK_MASS_PROFILE=official`。

### 分支是怎么来的

```
ours-813g-hd1910 (a790c92)
  + 5af629f  flamingo: commanded single-leg stand <-> two feet       ← 摘自 flamingo a99856c
  + 860778e  flamingo v2: swan-style backward leg lift                ← 020245b
  + 7cc9c7b  flamingo v3: no jammed joints, bounded head counterweight ← 5fb0be3
  + 82897fd  flamingo v3.1: close the kickstand exploit               ← 1feb0e9
  + 本文和 scripts/flamingo/
```

- 摘这 4 个提交时没有冲突，`tests/test_flamingo_cfg.py` 的 16 项测试全部通过。
- 目标姿态常数是在官方 737 g 模型上解出来的。换成 835 g 后，单腿姿态下整机质心相对支撑脚的偏移是：官方模型前 1.4 mm、左 7.1 mm；`ours_835g` 前 2.0 mm、左 7.0 mm。两者几乎一样，都在脚底范围内（脚底半宽 17.8 mm），所以**姿态常数不用改**。

## 2. 任务定义（和 XL330 方案相同）

- **指令**：`twist = [flag, side, 0]`。`flag` 为 0 是双脚站、1 是单脚站；`side` 为 +1 是右脚撑地（左腿后抬）、−1 是左脚撑地。每种状态保持 3–6 s，单脚的概率是 50 %。
- **观测和动作**：观测 61 维，动作 14 维，目标位置 = HOME + 1.0 × action，50 Hz。和 velstand / sitstand 是同一套接口。
- **目标姿态**：抬起的脚在支撑脚后方 73 mm、离地 119 mm，躯干侧倾 24.6°；每个关节离硬限位至少 0.08 rad。

**奖励**：

| 项 | 权重 | 作用 |
|---|---|---|
| `flamingo_composite` | 4.0 | 乘积项：接触 × 直立 × 姿态 × (质心 + (1−α)) |
| `flamingo_contact` | 2.0 | 支撑脚着地，抬起的脚离地 |
| `flamingo_swing_lift` / `flamingo_swing_behind` | 1.5 / 1.5 | 抬脚高度目标 0.08 m，在支撑脚后方目标 0.05 m；要求支撑脚着地、抬起的脚离地、躯干直立 |
| `flamingo_com` | 2.0 | 整机质心落在支撑脚上方，std 15 mm |
| `flamingo_upright` | 2.0 | 侧倾接近指令要求的角度 |
| `flamingo_pose` / `flamingo_height` / `flamingo_stillness` | 1.0 各 | 腿部姿态；躯干高度（双脚 116 mm，单脚 122 mm）；保持时静止 |
| `head_pose_tracking` | 0.3 | 头部指令跟踪，权重调低，让头可以当配重 |
| `non_sole_contact` | −2.0 | 脚底以外的部位碰地 |
| `joint_limits` / `action_over_limit` | −5.0 / −0.5 | 关节进入离硬限位 0.05 rad 的区域 / 目标位置超过限位 0.3 rad |
| `servo_stall` / `head_range` | −0.05 / −5.0 | 堵转（> 0.4 N·m 且速度 < 0.5 rad/s）/ 头部超出允许范围 |

**终止**：倾斜超过 50°，或超时（单回合 12 s）。

**课程**：

| 课程 | 安排 |
|---|---|
| 单脚出生概率 | 一开始 0.5，第 1500 次迭代降到 0.3 |
| 推力 | 迭代 1000：±0.025；1500：±0.05；2000：±0.1；2500：±0.15 m/s |

**PPO**：512-256-128 ELU，lr 1e-3，每个环境每次迭代 24 步，每 250 次迭代存一次检查点。

各版本的来由（v1 往前抬腿、v2 关节顶死、v3 撑脚架作弊、v3.1 修复）见 `flamingo` 分支的 `docs/flamingo_xl330.md`。

## 3. 现状：还没有在 HD1910 上训练过

目前最好的策略是 XL330 版的 `2026-10-30_11-23-07_flamingo_v31_ws` @ `model_1000`。把它直接放到 HD1910 + 835 g 上：

| 测试 | 结果 |
|---|---|
| 训练环境，无推力，16 个环境（`compare_ckpts.py`） | model_1000：13/16，侧倾 31°，最差关节力矩 p95 0.29 N·m，抖动 0.027 |
| 同上 | model_1500：13/16，0.69 N·m，0.057；model_3000：14/16，0.66 N·m，0.067 |
| `duck_runtime` 仿真，HD1910 + 835 g，6 个种子（`rehearse_runtime.py`） | 右脚撑地 1/6（左脚抬到约 67 mm，多数 2–3 s 后摔倒），左脚撑地 0/6（基本抬不起来） |
| 同上，HD1910 + 官方 737 g | 两侧都是 0/6 |
| 同上，官方 XL330 舵机 + 官方质量 | 右脚撑地 6/6，左脚撑地 2/6 |

结论是**换成 HD1910 舵机才是失败的主因**（质量都用官方的、只把 XL330 换成 HD1910，右脚撑地就从 6/6 掉到 0/6），要在 HD1910 上重新训练，或者在它上面微调。

建议的验收标准（我定的，可以改）：
- `compare_ckpts.py` 无推力 32/32；±0.15 推力下 ≥ 28/32。
- 最差关节力矩 p95 < 0.4 N·m（`servo_stall` 的阈值）。
- `rehearse_runtime.py` 左右两侧都 ≥ 5/6。

## 4. 训练命令（5090 工作站，仓库 `~/hl/Microduck/mdown`）

```bash
git fetch origin && git checkout flamingo-hd1910 && uv sync

# 1) 确认用的是 HD1910 + 835 g（训练启动时不会打印这两项）
uv run python -c "import mjlab_microduck.robot.microduck_constants as C, mjlab_microduck.robot.mass_profile as M; print(C.SERVO_MODEL, M.active_profile_name())"
# 应该输出：hd1910 ours_835g

# 2) 冒烟测试
MICRODUCK_WARM_START=1 uv run train Mjlab-Flamingo-Flat-Backlash-MicroDuck \
  --env.scene.num-envs 64 --agent.max-iterations 5 --agent.resume True \
  --agent.load-run 2026-10-30_11-23-07_flamingo_v31_ws --agent.load-checkpoint model_1000.pt \
  --agent.run-name smoke_hd1910

# 3) 正式训练：从 XL330 版的 model_1000 热启动
MICRODUCK_WARM_START=1 uv run train Mjlab-Flamingo-Flat-Backlash-MicroDuck \
  --env.scene.num-envs 4096 --agent.max-iterations 3000 --agent.resume True \
  --agent.load-run 2026-10-30_11-23-07_flamingo_v31_ws --agent.load-checkpoint model_1000.pt \
  --agent.run-name flamingo_hd1910
```

说明：

- **`--agent.load-run`** 指向 `logs/rsl_rl/microduck_flamingo/` 下 XL330 版的 run 目录。两个方案的 experiment 名都是 `microduck_flamingo`，日志放在同一个目录；`logs/` 不受 git 管，切分支不影响。
- **`MICRODUCK_WARM_START=1`**（`mdp.py:124-144`，Patch 5）会在加载权重后把计数清零，出生课程和推力课程从头开始。物理模型换了，先在没有推力的条件下适应 HD1910，再慢慢加推力。不加这个变量的话，迭代计数从 1000 接着算，推力在第 1 次迭代就到了 ±0.025，后面也会提前加大。
- **跑 3000 次迭代**是为了让推力走完整个课程（第 2500 次迭代到 ±0.15）。把第 1000、2000、3000 次的检查点拉回来，按第 5 节对比。
- **备选：从零训练。** 去掉 `MICRODUCK_WARM_START`、`--agent.resume` 和两个 `load-*` 参数，默认跑 6000 次迭代。这条路没有跑过。

## 5. 评估（`scripts/flamingo/`，在 CPU 上跑，用的是这个分支的 HD1910 模型）

```bash
# 选检查点：每个检查点输出一行，默认分别测无推力和 ±0.15 推力
uv run python scripts/flamingo/compare_ckpts.py logs/rsl_rl/microduck_flamingo/<run>/model_{1000,2000,3000}.pt
# 分场景细看：站住、抬腿、放下、空闲；还有每侧的成功率和抬脚高度
uv run python scripts/flamingo/eval_scenarios.py <ckpt> --push 0.15
# 录视频（服务器上需要 libosmesa6）
uv run python scripts/flamingo/record_video.py <ckpt> flamingo_hd1910.mp4
# 用机器上那套 runtime 预演：先导出并打包（第 6 节），再跑
uv run python scripts/flamingo/rehearse_runtime.py policies_flamingo --seeds 6
uv run python scripts/flamingo/rehearse_runtime.py policies_flamingo --supply robot   # 用这台机实测的电池压降
```

- `compare_ckpts` / `eval_scenarios` / `record_video` 用的是训练环境本身：去掉课程、关掉指令重采样，出生概率和推力用参数固定。这几个是判断策略好坏的主要依据。
- `rehearse_runtime` 用的是机器上那套 `duck_runtime` 代码：`Runner` / `Controller` / `policy_set`，清单解析、动作时长、收势、限位钳制都和真机一样，只把机体换成 SimIO。SimIO 没有齿隙，接触比训练环境硬，所以结果比训练环境保守。这一步用来上机前最后检查一遍。

## 6. 导出和部署

```bash
# 导出：观测归一化会打包进 ONNX。不要手工从 .pt 转换
uv run scripts/export.py Mjlab-Flamingo-Flat-Backlash-MicroDuck \
  --checkpoint-file logs/rsl_rl/microduck_flamingo/<flamingo_hd1910 的 run>/model_<N>.pt \
  --onnx-file flamingo.onnx --device cpu
uv run python scripts/flamingo/check_onnx.py <同一个 .pt> flamingo.onnx      # 输入输出 61 → 14，误差约 1e-6
uv run python scripts/flamingo/make_policy_set.py flamingo.onnx policies_flamingo --servos hd1910 --note "<run>@<N>"
```

`--servos` 只是清单里显示用的标签，runtime 会把它显示出来，但不会因为它改变任何行为。

**机器端**：

```bash
scp -r policies_flamingo user@microduck.local:~/microduck_rl/
# 在鸭子上：复制一份官方策略集，把 flamingo 的两个动作加到复制件里，原来的 policies/ 不动
python3 policies_flamingo/merge_into_official.py        # 生成 policies_official_flamingo/
duck duck_runtime.run --calib calib.json --set policies_official_flamingo --on-fall limp
```

- 按键：**4** = `flamingo_right`（右脚撑地，左腿后抬），**5** = `flamingo_left`。
- 单脚保持 6 s，然后双脚站 1.5 s 收势，再交回 velstand。
- 动作缩放 1.0，不加低通（清单里写了 `lowpass` 1.0，`duck_runtime` 会读这个字段），和训练时一样。
- 单脚动作执行期间，runtime 的摔倒预判是关掉的，所以第一次上机要用手扶着。
- 如果 `policies_official_flamingo/` 已经有了，只换里面的 `flamingo.onnx`，或者重跑合并脚本时加 `--force`。

## 7. 文件都在哪

| 什么 | 在哪 |
|---|---|
| 代码 | fork `sisjkly/microduck_rl`，分支 `flamingo-hd1910` |
| 训练日志和检查点 | 5090：`~/hl/Microduck/mdown/logs/rsl_rl/microduck_flamingo/<run>/` |
| 拉回来的检查点 | Mac：`~/ckpts/<run>/` |
| 当前部署包 | Mac：`~/ckpts/policies_flamingo/`。**这里面还是 XL330 版的 model_1000**，HD1910 版训好后替换 |
| 配置测试 | `tests/test_flamingo_cfg.py`（16 项） |
