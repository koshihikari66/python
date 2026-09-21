import math
import time

import pigpio

# ── 핀 / PWM 설정 ───────────────────────────────────────────
YAW_PIN = 23
PITCH_PIN = 15

PW_MIN = 500
PW_MID = 1500
PW_MAX = 2500

YAW_MIN, YAW_MAX = 0, 180
PITCH_MIN, PITCH_MAX = 90, 180


class PIDController:
    """단일 축 PID 컨트롤러.

    현재 설정에서 I 게인은 0이므로 I항과 적분 관련 상태는 제거했다.
    yaw/pitch는 이제 서로 다른 게인을 가질 수 있고, D항의 이전 오차
    상태도 축별로 독립적이어야 하므로 PIDController 인스턴스는
    ServoController에서 축별로 각각 생성해 사용한다.
    """

    def __init__(
        self,
        kp: float,
        kd: float,
        dt: float = 1 / 30,
        output_limit: float = 2.0,
        deadband: float = 1.0,
    ):
        self.kp = kp
        self.kd = kd
        self.dt = dt
        self.output_limit = output_limit
        self.deadband = deadband
        self._prev_error = 0.0

    def compute(
        self,
        error: float,
        velocity: float = 0.0,
        use_d: bool = True,
    ) -> float:
        if abs(error) < self.deadband:
            error = 0.0

        p = self.kp * error

        if not use_d:
            d = 0.0
        elif abs(velocity) > 1e-6:
            d = -self.kd * velocity
        else:
            d = self.kd * (error - self._prev_error) / self.dt

        self._prev_error = error
        output = p + d
        return max(-self.output_limit, min(self.output_limit, output))

    def reset(self):
        self._prev_error = 0.0


class ServoController:
    """yaw/pitch가 각각 독립된 PID 게인을 갖는 2축 서보 컨트롤러.

    두 축의 물리적 특성(부하, 기어비, 마찰 등)이 다를 수 있으므로
    kp/kd/output_limit/deadband를 축별로 따로 튜닝할 수 있게 했다.

    현재 서보 각도는 "1차 지연 + 슬루율 제한" 모델로 추정한다.
      - tau       : 서보(부하 포함)의 시정수 [s]. 명령각을 지수적으로 뒤따라간다.
      - max_speed : 서보의 최대 각속도 [deg/s]. 큰 오차에서 속도 상한으로 작용한다.

    모델각(yaw_angle/pitch_angle)이 명령각(*_cmd_angle)보다 늦게 따라가므로
    move()의 `모델각 ∓ delta`는 명령을 무한정 누적하지 않고, 실제 서보가 아직
    도달하지 못한 분량만큼만 앞서 나간다. 대신 프레임당 실제 이동량이
    kp*err의 alpha(=1-exp(-dt/tau))배가 되므로, 기존 kp/kd로 튜닝한 공칭 속도를
    유지하려면 kp/kd를 1/alpha배, output_limit을 max_speed*dt/alpha로 잡는다.
    (split.py에서 자동 계산한다.)
    """

    def __init__(
        self,
        yaw_pin: int = YAW_PIN,
        pitch_pin: int = PITCH_PIN,
        yaw_kp: float = 1.1,
        yaw_kd: float = 0.011,
        yaw_output_limit: float = 4.0,
        yaw_deadband: float = 2.0,
        pitch_kp: float = 1.1,
        pitch_kd: float = 0.011,
        pitch_output_limit: float = 4.0,
        pitch_deadband: float = 2.0,
        dt: float = 1 / 30,
        max_speed: float = 90.0,
        tau: float = 0.10,
        home_step_deg: float = 5.0,
        home_step_delay: float = 0.06,
    ):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError(
                "pigpiod가 실행 중이 아닙니다. "
                "'sudo pigpiod'를 먼저 실행하세요."
            )

        self.yaw_pin = yaw_pin
        self.pitch_pin = pitch_pin
        self.dt = dt
        self.max_speed = max(1e-6, max_speed)
        self.tau = max(1e-6, tau)
        # 이산 1차 지연의 프레임당 수렴 비율: 0 < alpha < 1
        self._alpha = 1.0 - math.exp(-dt / self.tau)

        self.yaw_angle = 90.0
        self.pitch_angle = 90.0
        self.yaw_cmd_angle = 90.0
        self.pitch_cmd_angle = 90.0

        self.home_step_deg = home_step_deg
        self.home_step_delay = home_step_delay

        # 축별로 게인이 다르고, 이전 오차 상태도 섞이면 안 되므로
        # PIDController 인스턴스를 축별로 독립 생성한다.
        self.yaw_pid = PIDController(
            kp=yaw_kp,
            kd=yaw_kd,
            dt=dt,
            output_limit=yaw_output_limit,
            deadband=yaw_deadband,
        )
        self.pitch_pid = PIDController(
            kp=pitch_kp,
            kd=pitch_kd,
            dt=dt,
            output_limit=pitch_output_limit,
            deadband=pitch_deadband,
        )

        self._write_yaw_cmd(90.0)
        self._write_pitch_cmd(90.0)
        time.sleep(0.5)

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _angle_to_pw(self, angle_deg: float) -> int:
        pw = PW_MID + ((angle_deg - 90) / 180.0) * (PW_MAX - PW_MIN)
        return int(max(PW_MIN, min(PW_MAX, pw)))

    def _set_pw(self, pin: int, pw: int):
        self.pi.set_servo_pulsewidth(pin, pw)

    def _lag_toward(self, current: float, target: float) -> float:
        """1차 지연 + 슬루율 제한으로 모델각을 한 프레임 진행시킨다.

        step = (target - current) * alpha       # 지수적으로 명령각을 뒤따름
        |step| <= max_speed * dt                # 큰 오차에서는 속도 상한
        """
        step = (target - current) * self._alpha
        max_step = self.max_speed * self.dt
        step = max(-max_step, min(max_step, step))
        return current + step

    def _update_servo_position(self):
        self.yaw_angle = self._lag_toward(
            self.yaw_angle,
            self.yaw_cmd_angle,
        )
        self.pitch_angle = self._lag_toward(
            self.pitch_angle,
            self.pitch_cmd_angle,
        )

    def _write_yaw_cmd(self, angle_deg: float):
        self.yaw_cmd_angle = self._clamp(angle_deg, YAW_MIN, YAW_MAX)
        self._set_pw(self.yaw_pin, self._angle_to_pw(self.yaw_cmd_angle))

    def _write_pitch_cmd(self, angle_deg: float):
        self.pitch_cmd_angle = self._clamp(angle_deg, PITCH_MIN, PITCH_MAX)
        self._set_pw(self.pitch_pin, self._angle_to_pw(self.pitch_cmd_angle))

    def move(
        self,
        yaw_err: float,
        pitch_err: float,
        vx_kalman: float = 0.0,
        vy_kalman: float = 0.0,
        use_d: bool = True,
    ):
        yaw_delta = self.yaw_pid.compute(
            yaw_err,
            velocity=vx_kalman,
            use_d=use_d,
        )
        pitch_delta = self.pitch_pid.compute(
            pitch_err,
            velocity=vy_kalman,
            use_d=use_d,
        )

        self._write_yaw_cmd(self.yaw_angle - yaw_delta)
        self._write_pitch_cmd(self.pitch_angle + pitch_delta)
        self._update_servo_position()

    def _ramp_to(
        self,
        target_yaw: float,
        target_pitch: float,
        step_deg: float,
        step_delay: float,
    ):
        start_yaw = self.yaw_cmd_angle
        start_pitch = self.pitch_cmd_angle

        dist = max(
            abs(target_yaw - start_yaw),
            abs(target_pitch - start_pitch),
        )
        if dist < 1e-6:
            return

        steps = max(1, math.ceil(dist / step_deg))
        for i in range(1, steps + 1):
            frac = i / steps
            self._write_yaw_cmd(
                start_yaw + (target_yaw - start_yaw) * frac
            )
            self._write_pitch_cmd(
                start_pitch + (target_pitch - start_pitch) * frac
            )
            time.sleep(step_delay)

    def center(self):
        self.yaw_pid.reset()
        self.pitch_pid.reset()

        self._write_yaw_cmd(90.0)
        self._write_pitch_cmd(90.0)
        self.yaw_angle = 90.0
        self.pitch_angle = 90.0

    def stop(self):
        self._ramp_to(
            90.0,
            90.0,
            step_deg=self.home_step_deg,
            step_delay=self.home_step_delay,
        )
        self.center()
        time.sleep(1)

        self._set_pw(self.yaw_pin, 0)
        self._set_pw(self.pitch_pin, 0)
        self.pi.stop()