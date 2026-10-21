// LSM6DSV16X -> ESP32 -> Pi: the official body-IMU block over a UART.
//
// The sensor is configured exactly like the official imu_to_dxl board / kissqy's FT6
// firmware (imu-source/src/main.c imu_init): XL +-4 g, GY +-500 dps, XL/GY ODR 120 Hz,
// SFLP 120 Hz game rotation batched to FIFO (stream mode), zero initial gyro bias.
// Each frame carries the same 12-byte block the official runtime decodes
// (duck-control/src/imu.rs): gyro x/y/z raw i16 LE + the last SFLP game-rotation FIFO
// word (quaternion x/y/z as IEEE half). Host side: duck_runtime/esp32_link.py.
//
// Frame (17 bytes, every 5 ms):
//   0xA5 0x5A | seq u8 | flags u8 | block[12] | crc8(seq..block, poly 0x07)
//   flags bit0: a new quaternion arrived since the previous frame
//         bit1: SFLP has produced at least one quaternion since boot
//         bit2: sensor read error on this frame
//
// Wiring defaults (classic ESP32 DevKit) — change the #defines for S3/C3 boards:
//   IMU I2C: SDA 21, SCL 22 (SDO/SA0 high -> 0x6B, low -> 0x6A; auto-detected)
//   To Pi:   Serial2 TX 17 -> Pi GPIO15 (RXD), RX 16 <- Pi GPIO14 (TXD), common GND, 921600 8N1
//   Set LINK_USB 1 to send over the USB serial port instead (then Pi sees /dev/ttyUSB*/ttyACM*).

#include <Arduino.h>
#include <Wire.h>

extern "C" {
#include "lsm6dsv16x_reg.h"
}

#ifndef IMU_SDA
#define IMU_SDA 21
#endif
#ifndef IMU_SCL
#define IMU_SCL 22
#endif
#ifndef LINK_TX
#define LINK_TX 17
#endif
#ifndef LINK_RX
#define LINK_RX 16
#endif
#ifndef LINK_BAUD
#define LINK_BAUD 921600
#endif
#ifndef LINK_USB
#define LINK_USB 0
#endif
#define FRAME_PERIOD_US 5000
#define FIFO_DRAIN_LIMIT 8

static uint8_t imu_addr = 0x6B;
static stmdev_ctx_t ctx;
static uint8_t block[12];
static uint8_t seq = 0;
static bool sflp_seen = false;

#if LINK_USB
#define LINK Serial
#else
#define LINK Serial2
#endif

static int32_t i2c_write(void *, uint8_t reg, const uint8_t *buf, uint16_t len) {
  Wire.beginTransmission(imu_addr);
  Wire.write(reg);
  Wire.write(buf, len);
  return Wire.endTransmission() == 0 ? 0 : -1;
}

static int32_t i2c_read(void *, uint8_t reg, uint8_t *buf, uint16_t len) {
  Wire.beginTransmission(imu_addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return -1;
  if (Wire.requestFrom((int)imu_addr, (int)len) != len) return -1;
  for (uint16_t i = 0; i < len; ++i) buf[i] = Wire.read();
  return 0;
}

static void ms_delay(uint32_t ms) { delay(ms); }

static uint8_t crc8(const uint8_t *p, size_t n) {
  uint8_t c = 0;
  while (n--) {
    c ^= *p++;
    for (int b = 0; b < 8; ++b) c = (c & 0x80) ? (uint8_t)((c << 1) ^ 0x07) : (uint8_t)(c << 1);
  }
  return c;
}

static bool imu_init() {
  ctx.write_reg = i2c_write;
  ctx.read_reg = i2c_read;
  ctx.mdelay = ms_delay;
  ctx.handle = nullptr;
  ctx.priv_data = nullptr;

  uint8_t who = 0;
  const uint8_t candidates[2] = {0x6B, 0x6A};
  bool found = false;
  for (uint8_t a : candidates) {
    imu_addr = a;
    if (lsm6dsv16x_device_id_get(&ctx, &who) == 0 && who == LSM6DSV16X_ID) { found = true; break; }
  }
  if (!found) return false;

  const lsm6dsv16x_fifo_sflp_raw_t fifo_sflp = {.game_rotation = 1, .gravity = 0, .gbias = 0};
  const lsm6dsv16x_sflp_gbias_t zero_bias = {.gbias_x = 0.0f, .gbias_y = 0.0f, .gbias_z = 0.0f};
  return lsm6dsv16x_sw_por(&ctx) == 0 &&
         lsm6dsv16x_auto_increment_set(&ctx, PROPERTY_ENABLE) == 0 &&
         lsm6dsv16x_block_data_update_set(&ctx, PROPERTY_ENABLE) == 0 &&
         lsm6dsv16x_xl_full_scale_set(&ctx, LSM6DSV16X_4g) == 0 &&
         lsm6dsv16x_gy_full_scale_set(&ctx, LSM6DSV16X_500dps) == 0 &&
         lsm6dsv16x_fifo_watermark_set(&ctx, 1) == 0 &&
         lsm6dsv16x_fifo_sflp_batch_set(&ctx, fifo_sflp) == 0 &&
         lsm6dsv16x_fifo_mode_set(&ctx, LSM6DSV16X_STREAM_MODE) == 0 &&
         lsm6dsv16x_xl_data_rate_set(&ctx, LSM6DSV16X_ODR_AT_120Hz) == 0 &&
         lsm6dsv16x_gy_data_rate_set(&ctx, LSM6DSV16X_ODR_AT_120Hz) == 0 &&
         lsm6dsv16x_sflp_data_rate_set(&ctx, LSM6DSV16X_SFLP_120Hz) == 0 &&
         lsm6dsv16x_sflp_game_rotation_set(&ctx, PROPERTY_ENABLE) == 0 &&
         lsm6dsv16x_sflp_game_gbias_set(&ctx, &zero_bias) == 0;
}

// Same refresh as FT6 imu_refresh(): current gyro output + newest game-rotation word.
static bool imu_refresh(bool *error) {
  int16_t gyro[3];
  bool fresh_quat = false;
  *error = false;
  if (lsm6dsv16x_angular_rate_raw_get(&ctx, gyro) == 0) {
    for (int a = 0; a < 3; ++a) {
      block[2 * a] = (uint8_t)(gyro[a] & 0xFF);
      block[2 * a + 1] = (uint8_t)((uint16_t)gyro[a] >> 8);
    }
  } else {
    *error = true;
  }
  lsm6dsv16x_fifo_status_t st;
  if (lsm6dsv16x_fifo_status_get(&ctx, &st) == 0) {
    uint16_t n = st.fifo_level > FIFO_DRAIN_LIMIT ? FIFO_DRAIN_LIMIT : st.fifo_level;
    while (n--) {
      lsm6dsv16x_fifo_out_raw_t s;
      if (lsm6dsv16x_fifo_out_raw_get(&ctx, &s) != 0) { *error = true; break; }
      if (s.tag == (uint8_t)LSM6DSV16X_SFLP_GAME_ROTATION_VECTOR_TAG) {
        memcpy(block + 6, s.data, 6);
        fresh_quat = true;
        sflp_seen = true;
      }
    }
  } else {
    *error = true;
  }
  return fresh_quat;
}

void setup() {
  Serial.begin(115200);
#if !LINK_USB
  Serial2.begin(LINK_BAUD, SERIAL_8N1, LINK_RX, LINK_TX);
#else
  Serial.updateBaudRate(LINK_BAUD);
#endif
  Wire.begin(IMU_SDA, IMU_SCL, 400000);
  memset(block, 0, sizeof block);
  while (!imu_init()) {
#if !LINK_USB
    Serial.println("LSM6DSV16X not found / init failed, retrying");
#endif
    delay(500);
  }
}

void loop() {
  static uint32_t next = micros();
  bool error = false;
  bool fresh = imu_refresh(&error);
  uint8_t f[17];
  f[0] = 0xA5;
  f[1] = 0x5A;
  f[2] = seq++;
  f[3] = (fresh ? 1 : 0) | (sflp_seen ? 2 : 0) | (error ? 4 : 0);
  memcpy(f + 4, block, 12);
  f[16] = crc8(f + 2, 14);
  LINK.write(f, sizeof f);
  // A single I2C glitch is reported (flag bit2) and ridden out; only a sustained failure
  // (100 ms) re-initialises the chip, since a reset restarts the SFLP fusion.
  static uint8_t errors = 0;
  errors = error ? (uint8_t)min(errors + 1, 255) : 0;
  if (errors >= 20) {
    sflp_seen = false;
    memset(block, 0, sizeof block);
    imu_init();
    errors = 0;
  }
  next += FRAME_PERIOD_US;
  int32_t wait = (int32_t)(next - micros());
  if (wait > 0) delayMicroseconds(wait); else next = micros();
}
