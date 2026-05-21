import pigpio
import time
import numpy as np

# ── 핀 설정 ────────────────────────────────────────────────
YAW_PIN   = 23
PITCH_PIN = 15

PW_MIN  =  500
PW_MID  = 1500
PW_MAX  = 2500

YAW_MIN,   YAW_MAX   = 0, 180
PITCH_MIN, PITCH_MAX = 0, 180

# ── 1차 지연 모델 시정수 ───────────────────────────────────
# 실측 방법: 90°→150° 스텝 명령 후 실제 각도 로깅
# 37.9° (63.2%) 도달 시간을 측정 → 그게 τ
# 0.11sec/60deg 서보 기준 추정값: ~0.07s
SERVO_TAU = 0.07


# ── PID 컨트롤러 ───────────────────────────────────────────
class PIDController:
    """
    단일 축 PID 컨트롤러.

    Parameters
    ----------
    kp, ki, kd    : PID 게인
    dt            : 제어 주기 (초)
    output_limit  : 한 스텝당 최대 각도 변화량 [deg]
    integral_limit: 적분 와인드업 방지 한계값
    deadband      : 이 범위 내의 오차는 0으로 처리
    """

    def __init__(
        self,
        kp: float = 0.3,
        ki: float = 0.01,
        kd: float = 0.05,
        dt: float = 1 / 30,
        output_limit: float = 4.0,
        integral_limit: float = 20.0,
        deadband: float = 1.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limit   = output_limit
        self.integral_limit = integral_limit
        self.deadband       = deadband

        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self, error: float, velocity: float = 0.0) -> float:
        """
        PID 출력 계산.

        Parameters
        ----------
        error    : 현재 오차 [deg]
        velocity : 칼만 필터 추정 각속도 [deg/s].
                   제공 시 D항을 칼만 속도로 대체.
                   0이면 일반 미분 항 사용.

        Returns
        -------
        output : 서보 각도 증분 [deg]
        """
        if abs(error) < self.deadband:
            error = 0.0

        p = self.kp * error

        self._integral += error * self.dt
        self._integral = max(
            -self.integral_limit, min(self.integral_limit, self._integral)
        )
        i = self.ki * self._integral

        if abs(velocity) > 1e-6:
            d = -self.kd * velocity
        else:
            d = self.kd * (error - self._prev_error) / self.dt

        self._prev_error = error

        output = p + i + d
        return max(-self.output_limit, min(self.output_limit, output))

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0


# ── 서보 컨트롤러 (PID + 1차 지연 모델) ───────────────────
class ServoController:
    """
    YAW / PITCH 독립 PID 제어 서보 컨트롤러.

    1차 지연 모델로 실제 서보 위치를 추정합니다.
      actual += α × (commanded - actual)
      α = 1 - exp(-dt / τ)

    actual_yaw / actual_pitch 프로퍼티를 newredc.py에서
    월드 각도 계산에 사용하면 명령값과 실제값의 차이로 인한
    측정 오염을 줄일 수 있습니다.

    사용 예 (newredc.py):
        servo_yaw   = servo.actual_yaw    # 추정 실제 위치
        servo_pitch = servo.actual_pitch
        yaw_world   = servo_yaw   + yaw_rel
        pitch_world = servo_pitch + pitch_rel
    """

    def __init__(
        self,
        yaw_pin: int   = YAW_PIN,
        pitch_pin: int = PITCH_PIN,
        kp: float = 0.23,
        ki: float = 0.005,
        kd: float = 0.01 * np.pi / 180,
        dt: float = 1 / 30,
        output_limit: float   = 5.0,
        integral_limit: float = 30.0,
        deadband: float       = 0.5,
        servo_tau: float      = SERVO_TAU,
    ):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpiod가 실행 중이 아닙니다. 'sudo pigpiod'를 먼저 실행하세요.")

        self.yaw_pin   = yaw_pin
        self.pitch_pin = pitch_pin
        self.dt        = dt

        # 명령값 (commanded)
        self.yaw_angle   = 90.0
        self.pitch_angle = 90.0

        # 1차 지연 모델: 추정 실제값 (actual)
        self._actual_yaw   = 90.0
        self._actual_pitch = 90.0
        self._alpha = 1.0 - np.exp(-dt / servo_tau)

        pid_kwargs = dict(
            kp=kp, ki=ki, kd=kd,
            dt=dt,
            output_limit=output_limit,
            integral_limit=integral_limit,
            deadband=deadband,
        )
        self.yaw_pid   = PIDController(**pid_kwargs)
        self.pitch_pid = PIDController(**pid_kwargs)

        self._set_pw(self.yaw_pin,   PW_MID)
        self._set_pw(self.pitch_pin, PW_MID)
        time.sleep(0.5)

    # ── 내부 헬퍼 ──────────────────────────────────────────
    def _angle_to_pw(self, angle_deg: float) -> int:
        pw = PW_MID + ((angle_deg - 90) / 180.0) * (PW_MAX - PW_MIN)
        return int(max(PW_MIN, min(PW_MAX, pw)))

    def _set_pw(self, pin: int, pw: int):
        self.pi.set_servo_pulsewidth(pin, pw)

    def _update_actual(self):
        """
        1차 지연 모델로 추정 실제 위치 업데이트.
        move() 호출 시마다 실행 (= 매 제어 주기).

        α = 1 - exp(-dt/τ)
        actual += α × (commanded - actual)
        """
        self._actual_yaw   += self._alpha * (self.yaw_angle   - self._actual_yaw)
        self._actual_pitch += self._alpha * (self.pitch_angle - self._actual_pitch)

    # ── 추정 실제 위치 프로퍼티 ────────────────────────────
    @property
    def actual_yaw(self) -> float:
        """1차 지연 모델로 추정한 실제 yaw 각도 [deg]"""
        return self._actual_yaw

    @property
    def actual_pitch(self) -> float:
        """1차 지연 모델로 추정한 실제 pitch 각도 [deg]"""
        return self._actual_pitch

    # ── 단축 각도 설정 ─────────────────────────────────────
    def set_yaw(self, angle_deg: float):
        self.yaw_angle = max(YAW_MIN, min(YAW_MAX, angle_deg))
        self._set_pw(self.yaw_pin, self._angle_to_pw(self.yaw_angle))

    def set_pitch(self, angle_deg: float):
        self.pitch_angle = max(PITCH_MIN, min(PITCH_MAX, angle_deg))
        self._set_pw(self.pitch_pin, self._angle_to_pw(self.pitch_angle))

    # ── 메인 제어 인터페이스 ───────────────────────────────
    def move(
        self,
        yaw_err: float,
        pitch_err: float,
        vx_kalman: float = 0.0,
        vy_kalman: float = 0.0,
    ):
        """
        Parameters
        ----------
        yaw_err   : 수평 오차 [deg]
        pitch_err : 수직 오차 [deg]
        vx_kalman : 칼만 추정 yaw 각속도 [deg/s] (선택)
        vy_kalman : 칼만 추정 pitch 각속도 [deg/s] (선택)
        """
        yaw_cmd   = self.yaw_pid.compute(yaw_err,   velocity=vx_kalman)
        pitch_cmd = self.pitch_pid.compute(pitch_err, velocity=vy_kalman)

        self.set_yaw(self.yaw_angle   - yaw_cmd)
        self.set_pitch(self.pitch_angle + pitch_cmd)

        # 명령 적용 후 1차 지연 모델 업데이트
        self._update_actual()

    # ── 유틸리티 ───────────────────────────────────────────
    def center(self):
        """서보를 중앙(90°)으로 복귀하고 PID 및 실제값 추정 초기화."""
        self.yaw_pid.reset()
        self.pitch_pid.reset()
        self.set_yaw(90.0)
        self.set_pitch(90.0)
        self._actual_yaw   = 90.0
        self._actual_pitch = 90.0

    def stop(self):
        """PWM 신호 정지 및 pigpio 연결 해제."""
        self.set_yaw(90.0)
        self.set_pitch(90.0)
        self.pi.stop()