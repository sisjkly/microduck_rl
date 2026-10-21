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

## 硬件：本机四块板

```
                USB (micro-USB OTG)            TTL 半双工 1 Mbps
 Pi Zero 2W ───────────────────► FE-URT-2 ─────────────────► 15× HD1910（ID 见下）
   GPIO14 TXD ─► ESP32 RX(16)
   GPIO15 RXD ◄─ ESP32 TX(17)       I²C 400 kHz
   GND ──────── ESP32 GND   ESP32 ──────────────► LSM6DSV16X（固定在躯干上）
```

1. **舵机**：HD1910 ID 与官方一致：左腿 20–24，头颈 30–33，嘴 34，右腿 10–14；波特率 **1 Mbps**
   （飞特 FD 上位机设置）。FE-URT-2 自带方向切换；回显自动检测（`--echo auto`），
   串口打开时自动设低延迟模式。
2. **IMU = LSM6DSV16X，与官方 imu_to_dxl 板同一颗芯片**，由 ESP32 读取：
   `esp32_imu/` 固件用 ST 官方驱动、按官方/FT6 完全相同的配置（±4 g、±500 dps、120 Hz、
   SFLP game rotation 120 Hz）打出**与官方逐字节相同的 12 字节块**，经 UART 921600 发给 Pi，
   Pi 端 `esp32_link.py` 用同一个 `SflpDecoder` 解码。
   - **IMU 必须刚性固定在躯干上**（官方 IMU 板在躯干里）：策略观测的是躯干姿态，装在头上会跟着脖子转。
   - 朝向不用对齐，`tools imu-mount` 用两个姿态测出安装四元数。
   - 链路超时（帧 >50 ms 或四元数 >100 ms 没更新）或 ESP32 报传感器错误 → 拒绝启动；运行中则立即卸力。
3. **Pi 的 GPIO 串口**（给 ESP32 用，URT-2 占了唯一的 USB 口）：
   `sudo raspi-config` → Interface → Serial：login shell **否**、hardware **是**；
   `/boot/firmware/config.txt` 加 `dtoverlay=disable-bt`（把稳定的 PL011 串口给 GPIO14/15），重启后用 `/dev/serial0`。
   也可以给 Pi 接 USB Hub，把 ESP32 走 USB（固件 `-DLINK_USB=1`，端口变成 `/dev/ttyUSB*` 或 `/dev/ttyACM*`）。

### 烧录 ESP32
```bash
cd duck_runtime/esp32_imu
pio run -e esp32dev -t upload      # 按板子改 platformio.ini 的 board 和引脚
```
上电后用 `tools scan --imu-port /dev/serial0` 检查：应约 200 帧/秒、校验错误为 0、`ready=True`。

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
A="--port /dev/ttyUSB0 --imu-port /dev/serial0"
python -m duck_runtime.tools template calib.json          # 生成标定模板
python -m duck_runtime.tools scan $A                      # 15 个舵机应答 + IMU 链路 ~200 帧/秒
python -m duck_runtime.tools zero $A --calib calib.json       # 零位：腿伸直、头摆正
python -m duck_runtime.tools direction $A --calib calib.json  # 逐关节核对正方向
python -m duck_runtime.tools imu-mount $A --calib calib.json  # 直立 + 前倾两个姿态
python -m duck_runtime.tools live $A --calib calib.json       # 检查：角度、重力投影
python -m duck_runtime.tools home-test $A --calib calib.json  # 吊起，上力矩回 HOME
python -m duck_runtime.run --backend feetech $A --calib calib.json \
       --policy velstand.onnx --action-scale 0.7 --log first.npz   # 1 上力矩 → 2 启动策略
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

`tests/test_duck_runtime.py`（23 项）：观测布局、HOME、关节顺序与训练端一致；IMU 安装姿态往返；
控制器缩放/滤波语义；状态机（不请求不动、看门狗、跌倒卸力、倒地拒绝启动）；
ESP32 帧解析/校验/超时判定，以及经真实伪终端 200 Hz 推流；`FeetechIO` 对接一条**模拟的
HD1910 总线**（伪终端另一端按飞特协议应答，含回显/无回显两种）完成读、上力矩（P=5/D=0/
限 1000 回读校验）、写目标、卸力。ESP32 固件做了语法级编译（真机工具链在沙箱里装不上）。
