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
duck duck_runtime.calibrate --calib calib.json       # 零位：放进零位架子，逐个关节摆好→回车上力锁住→w 保存
duck duck_runtime.tools imu-mount --calib calib.json  # 姿态1 架子平放、姿态2 连架子前倾 30-60°（或 imu-snap）
duck duck_runtime.tools sign-test --calib calib.json  # 平地站：IMU 判膝/髋偏航方向，两项都应 RIGHT
duck duck_runtime.tools home-test --calib calib.json  # 吊起，上力矩 3 s 回 HOME，看每个关节误差
duck duck_runtime.run --calib calib.json --policy velstand.onnx --action-scale 0.7 --log walk.npz
#   键：1 上力矩回 HOME → 2 启动策略 → w/s a/d 给指令；3 停止回 HOME；0 卸力；Ctrl+C 退出
```

- 结果（walk7，2026-10-22）：连续行走 67 s 不倒；vx 0.2 直走偏航 +0.4°/s；
  wz +0.8 实际 +27.7°/s（约 60%，与 mjlab 仿真的跟踪比例一致）；动作 p99 0.74（仿真步态约 0.7）。
- `calibrate`（逐关节锁定标零位）：每个关节显示 **ID + 部位 + 名字** 和零位应该的样子；摆好按回车，
  这个舵机就上力保持（P=32，退出恢复），摆下一个时不会被碰偏。`+`/`-` 按编码器计数微调（0.09°），
  `u` 松开重摆，`f` 掰一下关节就告诉你是哪个 ID，`l` 看全部，`w` 保存（只改已锁定关节的 `zero_raw`，
  先自动备份 `calib.json.bak-时间`）。`--joints legs|left|right|head|名字` 只标一部分。
  零位变化超过 8° 会提示"是不是摆错关节"。
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
- 跌倒处理 `--on-fall`：
  - `limp`（默认）：倾斜 >60° 持续 200 ms 就卸力，不再动。
  - `getup`：**官方默认的"原地起身"**（robotd limp-fall）：预测到正在倒下就卸力软着地，角速度 <1 rad/s 持续 0.2 s
    （最多等 1.5 s）后上力矩，0.6 s 内摆回站姿，再交给 velstand v8 站起来。
    **另外（官方没有）**：倒着且静止（|角速度| < 0.3 rad/s）满 1 s 就重新摆一次站姿再交给步态，最多 3 次。
    原因：dash_20261027-071329.npz 两次侧躺 5 s 一动不动——官方 velstand 在“正侧躺 + 那个腿姿”有个不动点，
    输出恒定姿势；仿真从那一刻的精确状态出发同样卡死（截不截关节限位都一样），重摆一次就站起来
    （理想/满电 3/3）。limp-fall 管不到的两种情况靠它：躺着按 `2` 启动、动作中摔倒（动作里不做 limp-fall，同官方）。
  - `keep`（= `--keep-on-fall`）：一直驱动，让 v8 自己爬起来（纯官方行为，不重摆）。
  - keep/getup 下躺在地上也能按 `2` 启动；交回策略后 `--getup-timeout`（默认 5 s，每次重摆后重新计时）还没起来就卸力。
  - 仿真 835 g，被推倒 12 次：理想供电 getup 12/12、keep 11/12；**按本机现在的电池压降两者都只有 0–1/12**，
    同样线路换满电 8–9/12，满电且压降减半 12/12。起身是大力矩动作，先解决供电。
- 顶栏：连接状态、当前状态和网络、电池（舵机最低电压）、总电流、最高温度、控制频率、摔倒次数、红色卸力按钮。
- 电源/策略（上力矩、启动、停止）、速度和头部滑条、坐站和每个动作的按钮（带进度）、摔倒后卸力还是自己爬起来。
- 姿态仪（俯仰/横滚/角速度）、最近 20 秒曲线：姿态、电压和电流、策略输出、控制周期。
- 14 个关节：位置、目标、误差、速度、电流、电压、温度、在限位范围里的位置；超限变色。
- 记录：一键开始/停止，存到机器人上的 `logs/`（字段同 `--log`），页面里直接下载；事件列表（状态切换、摔倒、拒绝原因）。
- 键盘按键和终端版相同。关掉或切走页面 0.5 s 后速度指令自动归零（页面每 200 ms 发一次心跳）。
- 同一时间只能有一个程序占用舵机：用网页时不要再开 `duck_runtime.run`。
- 在电脑上试：`uv run python -m duck_runtime.dashboard --backend sim --set policies --supply robot`，打开 http://localhost:8000 。

### 嘴（第 15 个舵机，ID 15）

ESP32 桥接固件原本只管 ID 1–14，嘴的舵机从来没被读写过。三步接上：

1. **改固件并重刷**，两种方式任选：
   - **从 Pi 直接刷（SSH 里就行，不用拔线）**：把编译好的镜像 `bridge_mouth15.bin` 拷到鸭子上，然后
     `bash duck_runtime/esp32_bridge_patch/flash_bridge.sh bridge_mouth15.bin`。脚本会停掉占串口的服务、
     装 esptool、通过 ESP32-S3 自带的 USB 口自动进下载模式写入，最后打印桥服务的 ID。回退用 `bridge_stock14.bin`。
     镜像用 Arduino-ESP32 3.0.7 编译：ESP32S3 Dev Module、USB CDC On Boot = Enabled、Hardware CDC and JTAG、4 MB DIO。
   - **在电脑上用 Arduino IDE**：`python3 duck_runtime/esp32_bridge_patch/add_mouth.py /你的草图路径/microduck_esp32_bridge`
     （只改 `bridge_config.h`：舵机数 14→15，加入 ID 15，掩码 0x7FFF；原来 1–14 的配置原样保留，旧文件备份为
     `bridge_config.h.before_mouth`），再照之前的设置烧录。
2. **鸭子上**：`duck duck_runtime.tools scan` 应显示 `bridge serves IDs [1..15] (mouth firmware)`，ID 15 mode 4。
3. **标定嘴**：`duck duck_runtime.tools mouth --calib calib.json`：写入 ID 15 → 用手把嘴张开测方向 → 闭上嘴记零位（闭合 = -5°）。

之后网页中控「张嘴」滑条（-5° 闭 … 30°）或键 `m` 开合；终端版也是 `m`。嘴不是策略的关节，开合不影响走路。
捡东西（`g`）时嘴按训练的阶段自动开合：下蹲时张开到 25°，到底停留的 0.2 s 内咬合，起身时保持闭合，结束后回到滑条的位置。
官方策略本身不控制嘴（14 个输出里没有嘴，训练里下颌是被动的），官方 robotd 捡东西时也不张嘴；这段是我们加的。
旧固件 + 嘴未标定的 `calib.json` 照常工作；`calib.json` 写了 ID 15 但固件没刷，会提示先刷固件。

### 3. 官方策略集（坐下/站起、捡东西、踢球、翻滚）

`--set` 读官方 Hub 仓库 `pollen-robotics/microduck-policies` 的 `manifest.json`（schema 2），
逐项照官方 robotd（`control.rs:421-760`）调度，逻辑在 `policy_set.py` + `runner.py`：

| 键 | 动作 | 网络 | 指令 | 时长 | 动作幅度 |
|---|---|---|---|---|---|
| `x` | 坐下 / 站起 | alpha_sitstand | vx 槽=1 坐、0 站，头指令照常 | 坐稳 2 s 后才能站，站起 1 s | 1.0 |
| `g` | 捡东西 | alpha_ground_pick | [cos 2πφ, sin 2πφ, 0]，周期 4 s | 2.8 s | 1.0 |
| `k` / `l` | 左 / 右踢 | ball_kick_* | 全零 | 0.5 s | 1.0（站立幅度，见下） |
| `r` | 翻滚 | roulade | 全零 | 1.0 s | 1.0 |

- 一次只做一个动作，坐着时不能做其它动作；动作结束回到步态，指令归零。
- 踢腿和翻滚用 1.0 的动作幅度：官方源码注释写明"踢腿按站立幅度（1.0）调的"，但官方程序只在有独立站立网络时
  才用到它，velstand 没有，结果官方现在实际按步态的 0.9 跑。0.9 下右踢在本机摔倒（dash_20261023-104355.npz）；
  仿真按本机电池压降：0.9 摔 3/6，1.0 摔 0/6，左踢两者都不摔。
- 动作网络驱动多久再交回步态用 `--skill-duration 名字=秒` 改（可写多次，两套都生效），例如翻滚 3 s：
  `--skill-duration roulade=3`。翻滚的训练回合是 5 s（`microduck_roulade_env_cfg.py` `EPISODE_LENGTH_S`），
  翻过去、落地、站稳都学过，3 s 仍在训练范围内；官方 1 s 交回时身体还斜 ~70°。仿真（835 g，6 个种子，摔倒起身）：
  官方 理想供电 1 s 4/6 → 3 s 6/6，满电 0/6 → 1/6；kissqy 理想 6/6 → 6/6，满电 5/6 → 4/6；本机压降两套都 0/6。
  捡东西是相位指令，窗口 = 周期 × 0.7，拉长会让相位跑出训练范围，别改。
- 站起按官方固定 1 s 交给步态（`--rise-s` 可改）。试过"等躯干稳了再交接"：仿真里坐站网络被往后推时
  越扶越往后倒（-35°），按时交给步态反而救得回来，所以没采用。
- 动作结束后 2.5 s 内不因"跌倒"卸力：翻滚 1 s 交回时身体还斜 ~73°，velstand v8 要到 ~2.5 s 才自己翻正
  （仿真 835 g，8/8）。官方默认摔倒后继续驱动，所以不需要这一条。
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

### 4. 零指令站着一直晃（约 2.8 Hz）

现象（dash_1026 / set5 / set6 的零指令段）：机身前后晃，俯仰角速度标准差 3–7°/s，主峰 2.7–2.9 Hz；
踝关节目标来回抖约 ±1°，实际位置只跟 ±0.1–0.2°，膝盖不动。只保持 HOME（策略没启动）时 IMU 只有 0.05–0.4°/s，
所以是策略和真机舵机构成的闭环在振荡，不是传感器噪声。仿真（835 g、HD1910，再加 3–8 倍摩擦、到 100 ms 延迟、
2° 齿隙）都复现不出来：官方 velstand 是按 XL330 训练的，本机舵机和它不一样。

- kissqy 的“修复鸭身抖动”（control.43，`duck-control/src/hd1910.rs` 的 `write_fixed_settings`）只做了一件事：
  上力矩时把 RAM 里的 D 写成 0（地址 51），扭矩上限写成 1000（地址 48）。HD1910 出厂 RAM 是 P/D = 32/40，
  他之前的版本没写 D，所以 D=40 一直生效。**我们每次上力矩都写 P=5、D=0 并读回校验**（`bridge.py` `set_torque`），
  这一条早就有了。扭矩上限我们不写，用 `duck duck_runtime.tools regs` 看舵机上电后是多少（只读，站着也能跑）。
- kissqy 用的是他按 HD1910 自己训练的整套模型（velstand `Mjlab-VelStand-Rough-Backlash-MicroDuck`，P=5、820 g、
  2° 齿隙），本机实测（2026-10-26）零指令站立不晃。用法见下一节：**整套加载，不和官方模型混用**。

### 5. 两套模型：官方 和 kissqy（HD1910 自训），完全分开，随时切换

kissqy 整套都是按 HD1910 训练的（P=5、820 g、官方观测约定和 HOME，逐个核对过）：走路、坐站、捡、左右踢、翻滚。
在鸭子上一条命令下载（他仓库里的固件包，约 34 MB；六个模型都按 SHA-256 校验）并生成清单：

```bash
cd ~/microduck_rl && git pull
duck duck_runtime.fetch_set kissqy            # -> policies_hd1910/（manifest.json + 6 个 onnx）
duck duck_runtime.dashboard --calib calib.json --set policies --set policies_hd1910 --on-fall getup
```

- `--set` 给两次 = 两套都加载，第一个是启动时用的那套（上面是官方；想默认 kissqy 就把两个 `--set` 对调）。
  面板“动作”卡片里的“模型套”下拉框，或键 `p` 切换；终端版 `duck_runtime.run` 也是 `p`。
  **只有策略停止时能切**（先按 3），正在走的步态不会中途换网络。
- 两套完全分开：选了官方，走路、坐站、捡、踢、翻滚、**摔倒后翻身站起**全部是官方网络；选了 kissqy 全部是他的。
  摔倒起身 = 卸力 → 摆站姿 → 交给**当前这套的走路网络**翻身。
- 每套只用它自己的部署参数（清单里写着，`policy_set.py` 读）：
  官方 = 步态幅度 0.9、全局低通 0.5/0.7；kissqy = 步态幅度 1.0、低通 0.5/0.7，捡/踢/滚幅度 1.0 且**不加低通**
  （他的 V167 就这么跑），坐站走全局低通。
- 加载两套时 `--policy`、`--sitstand`、`--action-scale`、`--tuning trained` 都会被拒绝（会把两套混起来）；
  要试这些，只加载一套。`--skill-duration`（如 `roulade=3`）、`--rise-s` 对两套都生效。
- 仿真对比（835 g，6 个种子，`scripts/rehearse_set.py --on-fall getup`，“站稳”= 5 s 后直立）：

| 动作 | 理想供电 官方 / kissqy | 本机压降模型 官方 / kissqy | 满电 8.4 V（同样线路）官方 / kissqy |
|---|---|---|---|
| 坐下站起 | 6/6 / 6/6 | 6/6 / 6/6 | — |
| 捡东西 | 6/6 / 6/6 | 6/6 / 6/6 | — |
| 翻滚 | 4/6 / **6/6** | 0/6 / 0/6 | 0/6 / **5/6** |
| 左踢 | 6/6 / 6/6 | **6/6** / 0/6 | 6/6 / 6/6 |
| 右踢 | 6/6 / 6/6 | **6/6** / 0/6 | 6/6 / 5/6 |

  kissqy 的踢腿对掉压更敏感（给它加上全局低通也一样倒，所以不是滤波的事），翻滚比官方好。
  摔倒起身（站立中被推倒，12 s 内站回来，向后/向前/侧向各 4 次）：理想供电 官方 12/12（向后 ~2.0 s）/ kissqy 12/12
  （~2.6 s）；满电 7/12 / 6/12；本机压降 0/12 / 0/12，都卡在 ~40° 侧躺：起身成不成主要看供电。
- 他的模型只限个人非商业使用（见他仓库的 LICENSE）；`fetch_set` 不把模型放进本仓库，只在鸭子上下载。

## 先在电脑上预演

在电脑上（不是鸭子上）：`git clone -b ours-813g-hd1910 https://github.com/sisjkly/microduck_rl.git && cd microduck_rl && uv sync`
（uv 会装好 Python 3.12、MuJoCo、BAM 舵机模型；第一次较慢）。把鸭子上的 `policies/` 拷过来：
`scp -r user@microduck.local:~/microduck_rl/policies .`

```bash
uv run python -m duck_runtime.run --backend sim --policy policies/velstand.onnx --viewer   # 键盘开车，同真机按键
uv run scripts/rehearse_set.py --set policies                       # 每个动作 8 次，835 g + HD1910
uv run scripts/rehearse_set.py --set policies --scenario roulade --seeds 1 --viewer        # 看一次翻滚
uv run scripts/rehearse_set.py --set policies --scenario sitstand --push 0.35 --push-at 0.8  # 复现 set1 的站起摔倒
uv run scripts/rehearse_set.py --set policies --profile official    # 换成原版质量分布对比
```

**浏览器界面（mjlab 同款 viser）**：`uv run scripts/viser_duck.py --set policies`，打开 http://localhost:8080
（远程 GPU 机器：`ssh -L 8080:localhost:8080 <机器>`）。3D 场景和 Scene / Visualization / Groups 页是 mjviser 的
（相机跟随、接触点、接触力、碰撞体）；Duck 页：上力矩 / 启动 / 停止 / 卸力、重置、速度 1/4–1x、摔倒规则、
vx / vy / wz 和头部四个角度滑条、坐站和每个动作的按钮、推一下（dvx / dvy）、躯干俯仰/横滚和策略输出的实时曲线。
默认上力矩后自动启动策略（仿真里只上力矩不跑策略，鸭子会在 ~45 s 内慢慢往前倒）。

**电池压降**：`--supply robot` 用本机实测的压降（set6.npz：空闲 7.5 V，站起时最低约 4.6 V，踢腿 3–4 V；约 0.1 Ω），
`--supply robot-full` 同样线路换满电 8.4 V，默认 `ideal` 是训练时的设定（7.6 V，最低 6 V）。
用 `robot` 时仿真复现了真机现象：右踢往前倒、左踢没事，站起时膝盖晚动、伸直变慢；换 `robot-full` 右踢不倒。

`rehearse_set.py` 用的是和真机完全相同的 Runner / Controller / policy_set，只是身体换成 SimIO
（MuJoCo + BAM HD1910 + 质量配置，全碰撞场景）。其它参数：`--on-fall getup`（官方的原地起身）或 `keep`（一直驱动）、
`--cmd=0.05,0,-0.2`（全程走路指令）、`--policy` / `--sitstand`（换网络）、`--log DIR`（每次一个 npz，
字段同真机日志，可以直接对比）。macOS 上开 `--viewer` 要用 `mjpython` 代替 `python`。

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
