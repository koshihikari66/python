"""
mpu6050_imu.py
────────────────────────────────────────────────────────────
MPU6050 (I2C) pitch 전용 측정 모듈. bno085_imu.py(BNO085IMU)를 대체.

BNO085를 구할 수 없어 MPU6050으로 교체. BNO085와 달리 MPU6050은 칩 내부에
센서 퓨전(DMP)이 없으므로, 여기서는 소프트웨어 상보 필터(complementary
filter)로 아래 둘을 합성해 pitch를 추정한다.

  - 가속도계(중력 벡터) : 절대 기준이라 드리프트가 없지만, 진동/급가속
                          구간에서는 순간적으로 노이즈가 크다.
  - 자이로(각속도 적분) : 짧은 구간에서는 부드럽고 반응이 빠르지만,
                          계속 적분만 하면 드리프트가 무한정 누적된다.

상보 필터는 매 스텝마다 자이로 적분값을 가속도계 값 쪽으로 조금씩
끌어당기므로(comp_alpha로 비율 조절), bno085_imu.py의 설계 원칙과 같이
"오차가 무한정 누적되지 않는" 특성을 유지하면서도 순수 가속도계 값보다
훨씬 부드러운 pitch를 얻는다. use_gyro_fusion=False로 끄면 가속도계
값만 사용(BNO_REPORT_GRAVITY와 동일한 방식)한다.

adafruit-circuitpython + blinka 스택 대신 smbus2(순수 I2C 레지스터
접근, 매우 가벼움)만 사용해서 라즈베리파이 자원 부담을 최소화했다.

읽기 실패(I2C 노이즈, 타이밍 등) 시 예외를 던지지 않고 마지막 유효값을
반환하므로, 제어 루프(30Hz)에서 매 프레임 안전하게 호출할 수 있다.

필요 패키지 (Raspberry Pi에서 설치)
------------------------------------
    pip install smbus2

배선 / 준비
-----------
    - SDA/SCL을 라즈베리파이 I2C 핀에 연결 (raspi-config에서 I2C 활성화 필요)
    - I2C 주소는 AD0 핀 상태에 따라 0x68(기본, AD0=LOW) 또는 0x69(AD0=HIGH)
    - `i2cdetect -y 1` 로 주소 확인 가능

sccpid_first_order.py에서의 사용법은 bno085_imu.BNO085IMU와 동일한
공개 API(calibrate_zero, get_servo_pitch_angle, close)를 그대로 따른다.
"""

import time
import math

from smbus2 import SMBus

# ── MPU6050 레지스터 주소 ────────────────────────────────────
_PWR_MGMT_1   = 0x6B
_SMPLRT_DIV   = 0x19
_CONFIG       = 0x1A
_GYRO_CONFIG  = 0x1B
_ACCEL_CONFIG = 0x1C
_ACCEL_XOUT_H = 0x3B
_GYRO_XOUT_H  = 0x43

_ACCEL_SCALE_2G = 16384.0   # LSB / g        (±2g 범위 기준)
_GYRO_SCALE_250 = 131.0     # LSB / (deg/s)  (±250 deg/s 범위 기준)


class MPU6050IMU:
    """
    MPU6050 I2C 래퍼. pitch 절대각([deg])을 가속도계+자이로 상보 필터로
    제공한다. BNO085IMU와 동일한 인터페이스를 유지한다.
    """

    def __init__(
        self,
        i2c_addr: int = 0x68,
        i2c_bus: int = 1,
        use_gyro_fusion: bool = True,   # 상보 필터(자이로+가속도) 사용 여부
        comp_alpha: float = 0.98,       # 상보 필터 계수 (자이로 비중, 0~1)
        # 서보 장착 방향에 따라 부호가 반대일 수 있으므로 부호 반전 옵션 제공.
        # 실측 후 부호가 반대로 움직이면 -1.0으로 바꾼다.
        pitch_sign: float = 1.0,
    ):
        self.bus  = SMBus(i2c_bus)
        self.addr = i2c_addr

        # 절전모드 해제
        self.bus.write_byte_data(self.addr, _PWR_MGMT_1, 0x00)
        time.sleep(0.05)
        # 샘플레이트: 1kHz / (1 + 4) = 200Hz
        self.bus.write_byte_data(self.addr, _SMPLRT_DIV, 0x04)
        # DLPF ~44Hz (진동/고주파 노이즈 억제)
        self.bus.write_byte_data(self.addr, _CONFIG, 0x03)
        # 자이로 ±250 deg/s, 가속도 ±2g (둘 다 이 용도엔 충분히 정밀)
        self.bus.write_byte_data(self.addr, _GYRO_CONFIG, 0x00)
        self.bus.write_byte_data(self.addr, _ACCEL_CONFIG, 0x00)

        self.use_gyro_fusion = use_gyro_fusion
        self.comp_alpha      = comp_alpha
        self.pitch_sign      = pitch_sign

        # 읽기 실패 시 폴백으로 쓸 마지막 유효값
        self._last_pitch = 0.0
        self._last_t     = None

        # calibrate_zero()로 설정되는 0점 보정값 (서보 90도 기준 자세)
        self._pitch_offset = 0.0

        # 상보 필터 시작점을 가속도계 값으로 초기화
        try:
            self._last_pitch = self._read_accel_pitch()
        except Exception:
            self._last_pitch = 0.0
        self._last_t = time.time()

    # ── 내부: 레지스터 읽기 ──────────────────────────────
    @staticmethod
    def _to_signed16(hi: int, lo: int) -> int:
        val = (hi << 8) | lo
        return val - 65536 if val >= 32768 else val

    def _read_raw(self, reg: int):
        data = self.bus.read_i2c_block_data(self.addr, reg, 6)
        x = self._to_signed16(data[0], data[1])
        y = self._to_signed16(data[2], data[3])
        z = self._to_signed16(data[4], data[5])
        return x, y, z

    def _read_accel_pitch(self) -> float:
        ax, ay, az = self._read_raw(_ACCEL_XOUT_H)
        ax /= _ACCEL_SCALE_2G
        ay /= _ACCEL_SCALE_2G
        az /= _ACCEL_SCALE_2G
        return math.degrees(math.atan2(-ax, math.hypot(ay, az)))

    def _read_gyro_y_rate(self) -> float:
        # bno085_imu.py의 pitch 축 정의(atan2(-ax, ...))에 대응하는 각속도는
        # 자이로 Y축이다.
        _, gy, _ = self._read_raw(_GYRO_XOUT_H)
        return gy / _GYRO_SCALE_250   # [deg/s]

    # ── 공개 API ─────────────────────────────────────────
    def read_pitch(self) -> float:
        """
        원시 pitch(0점 보정 전) [deg] 반환.
        실패 시 마지막 유효값을 반환한다.
        """
        now = time.time()
        dt  = (now - self._last_t) if self._last_t else 0.0
        self._last_t = now

        try:
            accel_pitch = self._read_accel_pitch()

            # dt가 비정상적으로 크면(첫 호출, 프레임 드랍 등) 자이로 적분을
            # 건너뛰고 가속도계 값을 그대로 사용해 오버슈트를 방지한다.
            if self.use_gyro_fusion and 0.0 < dt < 0.5:
                gyro_rate  = self._read_gyro_y_rate()
                gyro_pitch = self._last_pitch + gyro_rate * dt
                pitch = (
                    self.comp_alpha * gyro_pitch
                    + (1.0 - self.comp_alpha) * accel_pitch
                )
            else:
                pitch = accel_pitch

            self._last_pitch = pitch
            return pitch
        except Exception:
            return self._last_pitch

    def calibrate_zero(self, samples: int = 30, delay: float = 0.02):
        """
        현재 자세를 0점으로 잡는다.
        서보가 실제로 명령각 90도(수평 기준 자세)에 도달해 안정된 상태에서
        호출해야 이후 get_servo_pitch_angle()의 90도 기준이 정확해진다.
        평균으로 노이즈를 줄인다.
        """
        vals = []
        for _ in range(samples):
            vals.append(self.read_pitch())
            time.sleep(delay)
        self._pitch_offset = sum(vals) / len(vals)
        return self._pitch_offset

    def get_servo_pitch_angle(self):
        """
        sccpid_first_order.py의 각도 관례(PITCH_MIN=90 ~ PITCH_MAX=180,
        90=수평 기준)에 맞춰 IMU 측정 pitch를 변환해서 반환한다.

        calibrate_zero() 시점 대비 상대 변화량을 90도에 더하는 방식이므로,
        절대 자세가 아니라 "0점 보정 기준 상대각 + 90"이라는 점에 유의한다.
        """
        pitch = self.read_pitch()
        return 90.0 + self.pitch_sign * (pitch - self._pitch_offset)

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass