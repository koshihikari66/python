"""
bno085_imu.py
─────────────
BNO085 (I2C) 절대 자세 측정 모듈.

설계 근거
---------
서보 각도의 "현재 실제 각도"를 선형(속도 제한) 모델로 추정하는 대신,
IMU로 직접 측정한 값을 쓰면 오차 누적 없이 실제 각도를 알 수 있다.
다만 축마다 신뢰할 수 있는 소스가 다르다:

  - Pitch/Roll : 중력 벡터(Gravity)만으로 절대각을 구할 수 있고,
                 자이로 적분 드리프트나 지자기 간섭의 영향을 받지 않는다.
                 → 기본값으로 BNO_REPORT_GRAVITY 사용 (가속도계 기반 융합치).
  - Yaw        : 절대 기준이 지자기(마그네토미터)뿐이라 모터/금속 근접 시
                 간섭에 취약하다. 이 프로젝트는 현재 yaw를 90도로 고정해서
                 구동하지 않으므로, yaw는 이 모듈에서 기본적으로 사용하지 않는다.
                 필요해지면 use_rotation_vector=True로 켜서 quaternion 기반
                 yaw/pitch/roll을 모두 얻을 수 있다(단, yaw는 지자기 영향을 받음).

읽기 실패(I2C 노이즈, 타이밍 등) 시 예외를 던지지 않고 마지막 유효값을
반환하므로, 제어 루프(30Hz)에서 매 프레임 안전하게 호출할 수 있다.

필요 패키지 (Raspberry Pi에서 설치)
------------------------------------
    pip install adafruit-circuitpython-bno08x adafruit-blinka

배선 / 준비
-----------
    - SDA/SCL을 라즈베리파이 I2C 핀에 연결 (raspi-config에서 I2C 활성화 필요)
    - I2C 주소는 ADR 핀 상태에 따라 0x4A(기본, ADR=LOW) 또는 0x4B(ADR=HIGH)
    - `i2cdetect -y 1` 로 주소 확인 가능
"""

import time
import math

import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_GRAVITY,
    BNO_REPORT_ROTATION_VECTOR,
)


class BNO085IMU:
    """
    BNO085 I2C 래퍼. pitch/roll 절대각을 [deg] 단위로 제공한다.
    """

    def __init__(
        self,
        i2c_addr: int = 0x4A,
        i2c_freq: int = 400_000,
        use_rotation_vector: bool = False,
        # 서보 장착 방향에 따라 부호가 반대일 수 있으므로 부호 반전 옵션 제공.
        # 실측 후 부호가 반대로 움직이면 -1.0으로 바꾼다.
        pitch_sign: float = 1.0,
    ):
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=i2c_freq)
        self.bno = BNO08X_I2C(self.i2c, address=i2c_addr)

        self.use_rotation_vector = use_rotation_vector
        if use_rotation_vector:
            self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
        else:
            self.bno.enable_feature(BNO_REPORT_GRAVITY)

        self.pitch_sign = pitch_sign

        # 읽기 실패 시 폴백으로 쓸 마지막 유효값
        self._last_pitch = 0.0
        self._last_roll  = 0.0

        # calibrate_zero()로 설정되는 0점 보정값 (서보 90도 기준 자세)
        self._pitch_offset = 0.0

    # ── 내부: 원시 pitch/roll 계산 ──────────────────────────
    def _read_gravity_pitch_roll(self):
        gx, gy, gz = self.bno.gravity  # [m/s^2], 센서 프레임 기준 중력 벡터
        roll  = math.degrees(math.atan2(gy, gz))
        pitch = math.degrees(math.atan2(-gx, math.sqrt(gy * gy + gz * gz)))
        return pitch, roll

    def _read_rotation_vector_pitch_roll(self):
        qi, qj, qk, qr = self.bno.quaternion  # (i, j, k, real)

        sinr_cosp = 2 * (qr * qi + qj * qk)
        cosr_cosp = 1 - 2 * (qi * qi + qj * qj)
        roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        sinp = 2 * (qr * qj - qk * qi)
        sinp = max(-1.0, min(1.0, sinp))  # 부동소수 오차로 인한 domain error 방지
        pitch = math.degrees(math.asin(sinp))
        return pitch, roll

    # ── 공개 API ─────────────────────────────────────────
    def read_pitch_roll(self):
        """
        (pitch_deg, roll_deg) 원시값 반환. 실패 시 마지막 유효값을 반환한다.
        """
        try:
            if self.use_rotation_vector:
                pitch, roll = self._read_rotation_vector_pitch_roll()
            else:
                pitch, roll = self._read_gravity_pitch_roll()
            self._last_pitch = pitch
            self._last_roll  = roll
            return pitch, roll
        except Exception:
            return self._last_pitch, self._last_roll

    def calibrate_zero(self, samples: int = 30, delay: float = 0.02):
        """
        현재 자세를 0점으로 잡는다.
        서보가 실제로 명령각 90도(수평 기준 자세)에 도달해 안정된 상태에서
        호출해야 이후 get_servo_pitch_angle()의 90도 기준이 정확해진다.
        평균으로 노이즈를 줄인다.
        """
        vals = []
        for _ in range(samples):
            pitch, _ = self.read_pitch_roll()
            vals.append(pitch)
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
        pitch, _ = self.read_pitch_roll()
        return 90.0 + self.pitch_sign * (pitch - self._pitch_offset)

    def close(self):
        try:
            self.i2c.deinit()
        except Exception:
            pass