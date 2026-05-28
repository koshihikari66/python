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


# ── PID 컨트롤러 ───────────────────────────────────────────
class PIDController:
    """
    단일 축 PID 컨트롤러.
    D항은 오차 차분 기반 + LPF 적용.
    """

    def __init__(
        self,
        kp: float = 0.3,
        ki: float = 0.01,
        kd: float = 0.05,
        dt: float = 1 / 30,
        output_limit: float = 2.0,
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


        self._integral    = 0.0
        self._prev_error  = 0.0
        self._d_filtered  = 0.0

    def compute(self, error: float, velocity: float = 0.0, use_d: bool=True) -> float:
        """
        PID 출력 계산.
 
        Parameters
        ----------
        error    : 현재 오차 [deg]  (yaw_deg / pitch_deg)
        velocity : 칼만 필터 추정 속도 [px/frame].
                   제공 시 D항을 칼만 속도로 대체하여 더 부드러운 제어.
                   0이면 일반 미분 항 사용.
 
        Returns
        -------
        output : 서보 각도 증분 [deg]
        """
        # 데드밴드 처리
        if abs(error) < self.deadband:
            error = 0.0
 
        # P항
        p = self.kp * error
 
        # I항 (와인드업 방지)
        self._integral += error * self.dt
        self._integral = max(
            -self.integral_limit, min(self.integral_limit, self._integral)
        )
        i = self.ki * self._integral
 
        # D항: 칼만 속도가 있으면 활용, 없으면 오차 미분
        if not use_d:
            d = 0.0
        elif abs(velocity) > 1e-6:
            d = -self.kd * velocity
        else:
            d = self.kd * (error - self._prev_error) / self.dt
 
        self._prev_error = error
 
        output = p + i + d
        
        return max(-self.output_limit, min(self.output_limit, output))

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0


# ── 서보 컨트롤러 ─────────────────────────────────────────
class ServoController:
    """
    YAW / PITCH 독립 PID 제어 서보 컨트롤러.
    """

    def __init__(
        self,
        yaw_pin: int   = YAW_PIN,
        pitch_pin: int = PITCH_PIN,

        # 추천 초기값
        kp: float = 1.5,
        ki: float = 0.0,
        kd: float = 0.18,

        dt: float = 1 / 30,

        output_limit: float   = 5.0,
        integral_limit: float = 30.0,
        deadband: float       = 0.7,

        # 서보 명령 EMA 스무딩 (0=즉시반응, 1=변화없음)
        # 1-3프레임 주기 검출 진동 억제용 (권장: 0.5~0.7)
        cmd_smooth: float     = 0.08,
    ):
        self.pi = pigpio.pi()

        if not self.pi.connected:
            raise RuntimeError(
                "pigpiod가 실행 중이 아닙니다. "
                "'sudo pigpiod'를 먼저 실행하세요."
            )

        self.yaw_pin   = yaw_pin
        self.pitch_pin = pitch_pin

        self.yaw_angle   = 90.0
        self.pitch_angle = 90.0

        # EMA 스무딩 상태
        self.cmd_smooth    = cmd_smooth
        self._smooth_yaw   = 90.0
        self._smooth_pitch = 90.0

        pid_kwargs = dict(
            kp=kp,
            ki=ki,
            kd=kd,
            dt=dt,

            output_limit=output_limit,
            integral_limit=integral_limit,
            deadband=deadband
        )

        self.yaw_pid   = PIDController(**pid_kwargs)
        self.pitch_pid = PIDController(**pid_kwargs)

        self._set_pw(self.yaw_pin,   PW_MID)
        self._set_pw(self.pitch_pin, PW_MID)

        time.sleep(0.5)

    # ── 내부 헬퍼 ──────────────────────────────────────────
    def _angle_to_pw(self, angle_deg: float) -> int:
        pw = PW_MID + ((angle_deg - 90) / 180.0) * (PW_MAX - PW_MIN)

        return int(
            max(PW_MIN, min(PW_MAX, pw))
        )

    def _set_pw(self, pin: int, pw: int):
        self.pi.set_servo_pulsewidth(pin, pw)

    # ── 각도 설정 ─────────────────────────────────────────
    def set_yaw(self, angle_deg: float):
        self.yaw_angle = max(
            YAW_MIN,
            min(YAW_MAX, angle_deg)
        )

        self._set_pw(
            self.yaw_pin,
            self._angle_to_pw(self.yaw_angle)
        )

    def set_pitch(self, angle_deg: float):
        self.pitch_angle = max(
            PITCH_MIN,
            min(PITCH_MAX, angle_deg)
        )

        self._set_pw(
            self.pitch_pin,
            self._angle_to_pw(self.pitch_angle)
        )

    # ── 메인 제어 ─────────────────────────────────────────
    def move(
        self,
        yaw_err: float,
        pitch_err: float,
        vx_kalman: float = 0.0,
        vy_kalman: float = 0.0,
        use_d : bool=True
    ):
        """
        Parameters
        ----------
        yaw_err   : 수평 오차 [deg]  — xy2angle 출력값 그대로
        pitch_err : 수직 오차 [deg]  — xy2angle 출력값 그대로
        vx_kalman : 칼만 추정 x 속도 [px/frame] (선택)
        vy_kalman : 칼만 추정 y 속도 [px/frame] (선택)
        """
        yaw_cmd   = self.yaw_pid.compute(yaw_err,   velocity=vx_kalman, use_d=use_d)
        pitch_cmd = self.pitch_pid.compute(pitch_err, velocity=vy_kalman, use_d=use_d)

        raw_yaw   = self.yaw_angle   - yaw_cmd
        raw_pitch = self.pitch_angle + pitch_cmd

        # EMA 스무딩: 급격한 방향 전환 억제
        self._smooth_yaw   = (1 - self.cmd_smooth) * raw_yaw   + self.cmd_smooth * self._smooth_yaw
        self._smooth_pitch = (1 - self.cmd_smooth) * raw_pitch + self.cmd_smooth * self._smooth_pitch

        self.set_yaw(self._smooth_yaw)
        self.set_pitch(self._smooth_pitch)

    # ── 유틸리티 ──────────────────────────────────────────
    def center(self):
        self.yaw_pid.reset()
        self.pitch_pid.reset()

        self.set_yaw(90.0)
        self.set_pitch(90.0)

        self._smooth_yaw   = 90.0
        self._smooth_pitch = 90.0

    def stop(self):
        self.set_yaw(90.0)
        self.set_pitch(90.0)

        time.sleep(1)

        self._set_pw(YAW_PIN, 0)
        self._set_pw(PITCH_PIN, 0)

        self.pi.stop()
