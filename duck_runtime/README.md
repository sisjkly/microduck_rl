# duck_runtime — 官方策略 + HD1910 的精简运行端

用 **官方 ONNX 策略**（`pollen-robotics/microduck-policies`，如 `velstand.onnx`）驱动
**HD1910 + Raspberry Pi Zero 2W** 的 Microduck。官方 robotd 只支持 Dynamixel/XL330
和 Radxa ZERO 3W，这里只移植策略真正依赖的部分，约定逐行对照官方源码
（`pollen-robotics/microduck@942e58e`）：

| 内容 | 官方出处 | 这里 |
|---|---|---|
| 61 维观测、13 维指令、嘴部跳过 | `duck-control/src/obs.rs:12-256` | `contract.py` |
| HOME = 训练 HOME_FRAME | `duck-control/src/model.rs:37-53` | `contract.DEFAULT_POSITION`（测试锁定） |
| `目标 = HOME + 0.9·a`，头/腿低通 0.5/0.7，last_action 为原始输出 | `robotd/src/control.rs:654-735` | `controller.py` |
| IMU：SFLP 四元数 × 安装姿态⁻¹ → 重力投影，三点中值 | `duck-control/src/imu.rs` | `imu.py` |
| 安全：NaN 拒绝、限位、跌倒判定 gz>-0.5 持续 200 ms、指令看门狗 0.5 s | `duck-control/src/safety.rs` | `safety.py` |

控制器 / 安全 / 状态机对仿真和真机是**同一套代码**，只有最底层的 I/O 不同。

## 本机（最小可行路线）

```
Pi Zero 2W ──USB /dev/ttyACM0──► ESP32-S3（MDK2 桥接固件）──UART 单线 1 Mbps──► 14× HD1910（ID 1–14）
    └── I²C1 ──► LSM6DSV16X（0x6A/0x6B）
```

- **`bridge.py`**：对接 ESP32-S3 上现有的 MDK2 固件（`MicroDuck_XGO_r1/firmware`，带 NPD1）。
  每拍一次 SYNC READ 320（14 个关节的原始编码器、速度、电流、电压、温度、新鲜度），
  一次 SYNC WRITE 116 写目标。**标定全在 Python 里**：固件的 `center/dir` 映射被读回并反算，
  `bridge_config.h` 不用改。上力矩前用 NPD1 把 RAM P/D 设为训练值 **5/0** 并回读确认，退出时恢复原值。
  固件自带保护照常生效：主机 1 s 无通信、反馈 >100 ms、舵机错误、模式≠4 → 断力矩并锁存故障；
  全体断力矩（按 `0` 或 `tools relax`）清除故障。
- **`pi_imu.py`**：Pi 直接读 LSM6DSV16X，配置与官方 IMU 板相同（±4 g、±500 dps、120 Hz、
  SFLP game rotation 120 Hz 进 FIFO），拼成官方同款 12 字节块交给 `SflpDecoder`。
  四元数 >100 ms 不更新 → 拒绝启动 / 运行中卸力。
- **零依赖安装**：只用标准库 + numpy + onnxruntime，直接用鸭子上现成的 `~/microduck/.venv`。
- 关节 ID 来自鸭子上 `~/microduck/src/constants.py` 的接线；嘴没有舵机（标定文件里 `id: 0`）。
- **IMU 必须刚性固定在躯干上**；朝向不用对齐，`tools imu-mount` 测出来。

### 1. 装到鸭子上

```bash
cd ~ && git clone -b ours-813g-hd1910 https://github.com/sisjkly/microduck_rl.git
cd ~/microduck_rl
curl -L -o velstand.onnx https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/velstand.onnx
alias duck='~/microduck/.venv/bin/python -m'
sudo systemctl stop microduck-gamepad.service     # 它会占用 /dev/ttyACM0；想开机不自启再 disable
```

### 2. 上机流程（按顺序，一次只验证一件事；本机 2026-10-22 走通的版本）

```bash
duck duck_runtime.tools scan                          # 14 个舵机 ok、mode 4、NPD1 yes；IMU SFLP ≈120 Hz
duck duck_runtime.tools template calib.json           # 本机 ID 的标定模板（14 个关节方向都是 -1）
duck duck_runtime.tools zero      --calib calib.json  # 零位：用打印的零位架子（腿竖直、脚平、脖子竖直、头平）
duck duck_runtime.tools imu-mount --calib calib.json  # 姿态1 架子平放、姿态2 连架子前倾 30-60°（或 imu-snap）
duck duck_runtime.tools sign-test --calib calib.json  # 平地站：IMU 判膝/髋偏航方向，两项都应 RIGHT
duck duck_runtime.tools home-test --calib calib.json  # 吊起，上力矩 3 s 回 HOME，看每个关节误差
duck duck_runtime.run --calib calib.json --policy velstand.onnx --action-scale 0.7 --log walk.npz
#   键：1 上力矩回 HOME → 2 启动策略 → w/s a/d 给指令；3 停止回 HOME；0 卸力；Ctrl+C 退出
```

- 结果（walk7，2026-10-22）：连续行走 67 s 不倒；vx 0.2 直走偏航 +0.4°/s；
  wz +0.8 实际 +27.7°/s（约 60%，与 mjlab 仿真的跟踪比例一致）；动作 p99 0.74（仿真步态约 0.7）。
- 标定的坑（本机实测，每条都花过时间）：
  - **零位用架子**。手扶标出来的脖子零位偏了 24°（以为竖直其实已后仰），HOME 时头比官方靠后，
    整机重心偏后 → 一迈步就往后退；腿零位手扶也会把躯干倾斜算进髋零位。架子标完 14 个舵机都在 2048±140 内。
  - **方向别靠人眼**。`direction`（人掰关节）和 `pose-check`（看机器人动）的人眼判断两次与 IMU 相反；
    HOME 下接近 0 的膝、髋偏航、头偏航/横滚最容易错，用 `sign-test`（IMU 判）定。本机 14 个都是 -1
    （HD1910 与 XL330 转向相反，-1 正是这个补偿）。
  - **架子标零位后不要再用 `trim`**：它故意挪髋零位去凑站姿俯仰，策略看到的髋角就不准了
    （walk10：走 26 s 就失稳）。`trim` 只适合手扶标零位、零位本身有误差的情况。
  - **嘴别忘了装**：最初 813 g 是不带嘴（22 g）称的，模型里却有嘴，头的重心比模型靠后，
    走路一直往后靠；装上嘴（835 g）后走路俯仰 +0.6°，落在仿真范围（+0.2～+1.8°）内。质量配置已改为 `ours_835g`。
  - 一次一条命令，别整段粘贴（后面的行会被当成回车吃掉）；`--head-cmd=-10,...` 负数要用等号。
- `scan` 里如果某个舵机 `acc/time/speed` 非 0（`<-- move limit set`）先告诉我：桥只写目标位置，
  这几个寄存器里的旧值会限制每次运动，和训练假设不一致。
- 断力矩时关节若掰不动（减速比大），零位可用鸭子上已有的
  `~/microduck_fix_20260927/tools/calibrate_joint.py` 通电点动到位，记下 `raw_hd1910` 填进 `calib.json` 的 `zero_raw`。
- `run` 会拒绝未完成标定（每个关节 `calibrated`、`imu_mount_verified`）的配置。
- 第一次运行吊着、`--action-scale 0.7`；落地后再恢复默认 0.9。
- 摇杆量给到 **0.2–0.3** 才会迈步（策略本身的特性，仿真和 mjlab 一致）。
- 跌倒（倾斜 >60° 持续 200 ms）默认立即卸力；`--keep-on-fall` 改成继续驱动（官方默认行为）。

### 3. 官方策略集（坐下/站起、捡东西、踢球、翻滚）

`--set` 读官方 Hub 仓库 `pollen-robotics/microduck-policies` 的 `manifest.json`（schema 2），
逐项照官方 robotd（`control.rs:421-760`）调度，逻辑在 `policy_set.py` + `runner.py`：

| 键 | 动作 | 网络 | 指令 | 时长 | 动作幅度 |
|---|---|---|---|---|---|
| `x` | 坐下 / 站起 | alpha_sitstand | vx 槽=1 坐、0 站，头指令照常 | 坐稳 2 s 后才能站，站起 1 s | 1.0 |
| `g` | 捡东西 | alpha_ground_pick | [cos 2πφ, sin 2πφ, 0]，周期 4 s | 2.8 s | 1.0 |
| `k` / `l` | 左 / 右踢 | ball_kick_* | 全零 | 0.5 s | 跟步态（0.9） |
| `r` | 翻滚 | roulade | 全零 | 1.0 s | 跟步态 |

- 一次只做一个动作，坐着时不能做其它动作；动作结束回到步态，指令归零。
- 动作进行中或坐着时不因"跌倒"卸力（官方同样：翻滚本来就是倒过来的）。
- 没做的：每个动作的舵机增益比例（P 固定 5，NPD1 只能断力矩时改）、按住连续翻滚、轮滑（roller）相关的策略。
- `--policy X.onnx` 替换步态（如 Arena 的 sprint / hill-climb 参考策略），`--sitstand X.onnx` 替换坐站网络（如 got-a-talent）。
- 日志多了 `mode` 字段（walk / sit / rise / 技能名）。

```bash
mkdir -p ~/microduck_rl/policies && cd ~/microduck_rl/policies
for f in manifest.json velstand.onnx alpha_sitstand.onnx alpha_ground_pick.onnx ball_kick_left.onnx ball_kick_right.onnx roulade.onnx; do curl -fLO https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/$f; done
cd ~/microduck_rl
duck duck_runtime.run --calib calib.json --set policies --log set1.npz
```

## 先在电脑上预演

```bash
python -m duck_runtime.run --backend sim --policy velstand.onnx --viewer
```

## 其它接法（`--backend feetech`）

Pi 直连飞特总线（FE-URT / 官方 IMU 板 ID 200 / ESP32 IMU 串口链路 `--imu-port`）：`feetech.py`、
`esp32_link.py`、`esp32_imu/` 固件。本机不用。

## 已验证

仿真（官方 velstand v7，813 g + HD1910，mjlab 训练环境，8 env × 6 指令，`scripts/mjlab_policy_eval.py`）：

| 参数 | 前进 0.3 | 后退 −0.3 | 侧移 0.2 | 转向 1.0 | 跌倒 |
|---|---|---|---|---|---|
| 官方默认 0.9 + 低通 | 0.15 | −0.14 | 0.03 | 0.57 | 1/48 |
| 训练时 1.0 无滤波 | 0.15 | −0.18 | 0.05 | 0.60 | 1/48 |

`tests/test_bridge.py`：桥接后端对接一个**按 bridge_runtime.cpp 逻辑模拟的 MDK2 固件**（真实伪终端，
含非默认 center/dir 映射）——读数换算、上力矩时 P/D=5/0 回读且姿态不跳、目标逐计数正确、
故障拒绝上力矩与卸力清除、退出恢复原 P/D、端口独占、状态机回 HOME；IMU 驱动对接寄存器级假芯片——
配置与 ST 驱动一致（保留保留位）、重力/陀螺换算、就绪/超时/积压重置/I²C 错误、安装姿态。
**未在真机上跑过**：第一次请从 `scan` 开始，逐步把输出贴回来。
