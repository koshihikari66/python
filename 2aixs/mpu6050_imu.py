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
끌어당기므로(tau로 시간상수 조절), "오차가 무한정 누적되지 않는" 특성을
유지하면서도 순수 가속도계 값보다 훨씬 부드러운 pitch를 얻는다.
use_gyro_fusion=False로 끄면 가속도계 값만 사용한다.

── 논블로킹(백그라운드 스레드) 설계 ─────────────────────────
I2C read는 커널 스케줄링/버스 상황에 따라 가끔 수 ms 이상 튈 수 있는데,
이걸 카메라 캡처/검출/PID와 같은 스레드에서 매 프레임 동기 호출하면
그 튀는 순간이 그대로 프레임 처리 지연으로 이어져 프레임이 씹힐 수 있다.

angle_logger.py(AngleLogger)와 같은 원칙으로, 여기서도 전용 백그라운드
스레드가 자체 주기(poll_hz)로 계속 I2C를 읽어 상보 필터를 갱신하고
최신값만 캐시에 저장한다. 메인 루프(get_servo_pitch_angle 호출부)는
그 캐시를 읽기만 하므로 사실상 0비용이고, I2C read가 블로킹되는 동안에도
GIL이 풀리므로 메인 캡처 스레드를 막지 않는다.

가속도(6바이트)+자이로(6바이트)는 레지스터 주소가 연속(0x3B~0x48)이므로
따로 2번 읽지 않고 1번의 14바이트 블록 read로 합쳐 I2C 트랜잭션 자체도
줄였다.

상보 필터 계수는 고정 alpha 대신 tau(시간상수, 초)로 지정한다.
alpha = tau / (tau + dt) 로 매 주기 자동 계산되므로, poll_hz(폴링 속도)를
바꿔도 드리프트 억제 특성(tau)은 그대로 유지된다.

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
import threading

from smbus2 import SMBus

# ── MPU6050 레지스터 주소 ────────────────────────────────────
_PWR_MGMT_1     = 0x6B
_SMPLRT_DIV     = 0x19
_CONFIG         = 0x1A
_GYRO_CONFIG    = 0x1B
_ACCEL_CONFIG   = 0x1C
_ACCEL_XOUT_H   = 0x3B   # 가속도(6B) + 온도(2B) + 자이로(6B) = 연속 14바이트

_ACCEL_SCALE_2G = 16384.0   # LSB / g        (±2g 범위 기준)
_GYRO_SCALE_250 = 131.0     # LSB / (deg/s)  (±250 deg/s 범위 기준)

# ODR(출력 데이터 갱신 속도)는 SMPLRT_DIV=0x04 설정 기준 1kHz/(1+4)=200Hz.
# poll_hz가 이보다 빠르면 같은 값을 다시 읽는 것이므로 의미가 없다.
_MAX_USEFUL_POLL_HZ = 200.0


class MPU6050IMU:
    """
    MPU6050 I2C 래퍼. pitch 절대각([deg])을 가속도계+자이로 상보 필터로
    제공한다. BNO085IMU와 동일한 공개 인터페이스를 유지한다.

    내부적으로 전용 백그라운드 스레드가 poll_hz 주기로 I2C를 읽고 필터를
    갱신하며, 메인 스레드에서 부르는 read_pitch()/get_servo_pitch_angle()은
    캐시된 최신값만 반환한다(I2C 접근 없음, 논블로킹).
    """

    def __init__(
        self,
        i2c_addr: int = 0x68,
        i2c_bus: int = 1,
        use_gyro_fusion: bool = True,   # 상보 필터(자이로+가속도) 사용 여부
        tau: float = 1.5,               # 상보 필터 시간상수 [s] (자이로 바이어스 드리프트 억제 강도)
        poll_hz: float = 100.0,         # 백그라운드 스레드 I2C 폴링 주파수 [Hz]
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
        self.tau             = tau
        self.pitch_sign      = pitch_sign

        poll_hz = min(poll_hz, _MAX_USEFUL_POLL_HZ)
        self._poll_interval = 1.0 / poll_hz

        # calibrate_zero()로 설정되는 0점 보정값 (서보 90도 기준 자세)
        self._pitch_offset = 0.0

        # 캐시(락으로 보호) - 백그라운드 스레드가 쓰고, 메인 스레드가 읽는다.
        self._lock  = threading.Lock()
        self._cache_pitch = 0.0

        # 필터 내부 상태는 백그라운드 스레드만 접근하므로 락 불필요.
        self._filt_pitch = None
        self._last_t     = None

        # 시작 전 최초 1회 동기 read로 초기값을 채워둔다(빈 캐시로 시작 방지).
        try:
            self._filt_pitch = self._read_accel_pitch()
        except Exception:
            self._filt_pitch = 0.0
        self._cache_pitch = self._filt_pitch
        self._last_t = time.time()

        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    # ── 내부: 레지스터 읽기 ──────────────────────────────
    @staticmethod
    def _to_signed16(hi: int, lo: int) -> int:
        val = (hi << 8) | lo
        return val - 65536 if val >= 32768 else val

    def _read_accel_gyro(self):
        """가속도(6B)+온도(2B)+자이로(6B) 14바이트를 한 번의 I2C 트랜잭션으로 읽는다."""
        data = self.bus.read_i2c_block_data(self.addr, _ACCEL_XOUT_H, 14)
        ax = self._to_signed16(data[0],  data[1])
        ay = self._to_signed16(data[2],  data[3])
        az = self._to_signed16(data[4],  data[5])
        # data[6:8]는 온도, 사용 안 함
        gx = self._to_signed16(data[8],  data[9])
        gy = self._to_signed16(data[10], data[11])
        gz = self._to_signed16(data[12], data[13])
        return (ax, ay, az), (gx, gy, gz)

    def _read_accel_pitch(self) -> float:
        (ax, ay, az), _ = self._read_accel_gyro()
        ax /= _ACCEL_SCALE_2G
        ay /= _ACCEL_SCALE_2G
        az /= _ACCEL_SCALE_2G
        return math.degrees(math.atan2(-ax, math.hypot(ay, az)))

    # ── 백그라운드 폴링 스레드 ───────────────────────────
    def _poll_loop(self):
        while not self._stop.is_set():
            loop_start = time.time()
            self._poll_once()
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, self._poll_interval - elapsed))

    def _poll_once(self):
        now = time.time()
        dt  = (now - self._last_t) if self._last_t else self._poll_interval
        self._last_t = now

        try:
            (ax, ay, az), (gx, gy, gz) = self._read_accel_gyro()
            accel_pitch = math.degrees(
                math.atan2(-ax / _ACCEL_SCALE_2G,
                           math.hypot(ay / _ACCEL_SCALE_2G, az / _ACCEL_SCALE_2G))
            )

            # dt가 비정상적으로 크면(스레드 시작 직후, 일시적 스케줄링 지연 등)
            # 자이로 적분을 건너뛰고 가속도계 값을 그대로 사용해 오버슈트를 방지한다.
            if self.use_gyro_fusion and 0.0 < dt < 0.5:
                gyro_rate  = gy / _GYRO_SCALE_250   # pitch 축(atan2(-ax,...))에 대응하는 각속도 = gyro Y
                gyro_pitch = self._filt_pitch + gyro_rate * dt

                # alpha = tau / (tau + dt) : poll_hz(=dt)가 바뀌어도 tau(드리프트
                # 억제 시간상수)가 그대로 유지되도록 매 스텝 dt로부터 재계산.
                alpha = self.tau / (self.tau + dt)
                pitch = alpha * gyro_pitch + (1.0 - alpha) * accel_pitch
            else:
                pitch = accel_pitch

            self._filt_pitch = pitch
            with self._lock:
                self._cache_pitch = pitch
        except Exception:
            # I2C 읽기 실패 - 캐시를 마지막 유효값으로 유지(갱신하지 않음)
            pass

    # ── 공개 API ─────────────────────────────────────────
    def read_pitch(self) -> float:
        """
        원시 pitch(0점 보정 전) [deg] 반환. 백그라운드 스레드가 채워둔 캐시를
        읽기만 하므로 I2C 접근이 없고(논블로킹), 매 프레임 호출해도 안전하다.
        """
        with self._lock:
            return self._cache_pitch

    def calibrate_zero(self, samples: int = 30, delay: float = 0.02):
        """
        현재 자세를 0점으로 잡는다.
        서보가 실제로 명령각 90도(수평 기준 자세)에 도달해 안정된 상태에서
        호출해야 이후 get_servo_pitch_angle()의 90도 기준이 정확해진다.
        평균으로 노이즈를 줄인다. (delay 간격으로 캐시를 여러 번 샘플링 —
        poll_hz가 1/delay보다 빠르면 매번 새 값을 얻는다.)
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
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.bus.close()
        except Exception:
            pass