# duck_runtime — 官方策略 + HD1910 的精简运行端

用 **官方 ONNX 策略**（`pollen-robotics/microduck-policies`，如 `velstand.onnx`）驱动一台
**飞特 HD1910 + Raspberry Pi Zero 2W** 的 Microduck。官方 robotd 只支持 Dynamixel/XL330
和 Radxa ZERO 3W，这里只移植策略真正依赖的部分，约定逐行对照官方源码
（`pollen-robotics/microduck@942e58e`）：

| 内容 | 官方出处 | 这里 |
|---|---|---|
| 61 维观测、13 维指令、嘴部跳过 | `duck-control/src/obs.rs:12-256` | `contract.py` |
| HOME = 训练 HOME_FRAME | `duck-control/src/model.rs:37-53` | `contract.DEFAULT_POSITION`（测试锁定） |
| `目标 = HOME + 0.9·a`，头/腿低通 0.5/0.7，last_action 为原始输出 | `robotd/src/control.rs:654-735` | `controller.py` |
| IMU：SFLP 四元数 × 安装姿态⁻¹ → 重力投影，三点中值 | `duck-control/src/imu.rs` | `imu.py` |
| 安全：NaN 拒绝、限位、跌倒判定 gz>-0.5 持续 200 ms、指令看门狗 0.5 s | `duck-control/src/safety.rs` | `safety.py` |
| HD1910 寄存器（P=5、D=0、力矩上限 1000、目标 42、反馈 56） | kissqy `duck-control/src/{feetech,hd1910}.rs`（实机验证） | `feetech.py` |

仿真后端（`sim_io.py`，MuJoCo + BAM HD1910 + 813 g 质量配置）和真机后端走**同一套**
控制器/安全/状态机代码，IMU 甚至在字节级仿真（编码 → `SflpDecoder` 解码）。

## 硬件要求（先确认）

1. **舵机总线**：15 个 HD1910，ID 与官方一致：左腿 20–24，头颈 30–33，嘴 34，右腿 10–14；
   波特率 **1 Mbps**（用飞特 FD 上位机设置 ID 和波特率）。
2. **半双工转接**：需要自动方向切换（飞特 URT-1、微雪总线舵机转接板、或 74HC126 电路）。
   - USB 转接：FTDI 芯片要把 latency_timer 设成 1 ms（见下），否则每次读写多等 16 ms。
   - 若转接板会回显发送字节，运行时加 `--echo`。
3. **IMU**：`feetech.py` 默认 IMU 是总线上 **ID 200 的 RevA 板（STM32G031 + LSM6DSV16X）刷 FT6 固件**
   （kissqy `hardware/IMU-FIRMWARE.md`）。其他 IMU 通过 `FeetechIO(imu_source=...)` 接入，
   需提供躯干系角速度（rad/s）和重力投影（直立 = [0,0,-1]）。

## Pi Zero 2W 环境

```bash
# Raspberry Pi OS Lite 64-bit (Bookworm)
sudo apt install -y python3-venv git
git clone https://github.com/sisjkly/microduck_rl.git && cd microduck_rl && git checkout ours-813g-hd1910
python3 -m venv ~/duck && . ~/duck/bin/activate
pip install -r duck_runtime/requirements.txt
# USB 转接（FTDI）：把延迟降到 1 ms
echo 'ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"' | \
  sudo tee /etc/udev/rules.d/99-ftdi-latency.rules && sudo udevadm control --reload
sudo usermod -aG dialout $USER   # 串口权限，重新登录生效
```
策略文件：从 HF `pollen-robotics/microduck-policies` 下载 `velstand.onnx`（walk 槽，零指令即站立）。

## 上机流程（按顺序，一次只验证一件事）

```bash
P=/dev/ttyUSB0
python -m duck_runtime.tools template calib.json            # 生成标定模板
python -m duck_runtime.tools scan --port $P                 # 16 个设备都应应答
python -m duck_runtime.tools zero --port $P --calib calib.json       # 零位：腿伸直、头摆正
python -m duck_runtime.tools direction --port $P --calib calib.json  # 逐关节核对正方向
python -m duck_runtime.tools imu-mount --port $P --calib calib.json  # 直立 + 前倾两个姿态
python -m duck_runtime.tools live --port $P --calib calib.json       # 检查：角度、重力投影
python -m duck_runtime.tools home-test --port $P --calib calib.json  # 吊起，上力矩回 HOME
python -m duck_runtime.run --backend feetech --port $P --calib calib.json \
       --policy velstand.onnx --log first.npz                        # 1 上力矩 → 2 启动策略
```

- `run` 会拒绝未完成标定（`calibrated` / `imu_mount_verified`）的配置。
- 第一次运行建议吊着、加 `--action-scale 0.7`；落地后再恢复默认 0.9。
- 摇杆量给到 **0.2–0.3** 才会迈步（策略本身的特性，仿真和 mjlab 一致）。
- 跌倒（倾斜 >60° 持续 200 ms）默认立即卸力；`--keep-on-fall` 改成继续驱动（官方默认行为）。

## 先在电脑上预演

```bash
python -m duck_runtime.run --backend sim --policy velstand.onnx --viewer
```

## 已验证（仿真）

官方 velstand v7，813 g + HD1910（mjlab 训练环境，8 env × 6 指令，`scripts/mjlab_policy_eval.py`）：

| 参数 | 前进 0.3 | 后退 −0.3 | 侧移 0.2 | 转向 1.0 | 跌倒 |
|---|---|---|---|---|---|
| 官方默认 0.9 + 低通 | 0.15 | −0.14 | 0.03 | 0.57 | 1/48 |
| 训练时 1.0 无滤波 | 0.15 | −0.18 | 0.05 | 0.60 | 1/48 |

`tests/test_duck_runtime.py`：观测布局、HOME、关节顺序与训练端一致；IMU 安装姿态往返；
控制器缩放/滤波语义；飞特报文编解码；状态机（不请求不动、看门狗、跌倒卸力）。
