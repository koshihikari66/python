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

# 서보 PWM 갱신 주파수 (JX CLS6336HV 스펙: 1520µs / 330Hz)
# - pigpio의 set_servo_pulsewidth()는 50Hz 고정이라, 그 이상은 일반 PWM API로 구동한다.
# - pigpio는 정해진 주파수 표에서만 고르므로 330 요청 시 가장 가까운 값이 적용된다
#   (기본 샘플레이트 5µs 기준 320Hz). 실제 적용값은 시작할 때 출력된다.
# - 50 이하로 두면 기존 방식(set_servo_pulsewidth, 50Hz)으로 동작한다.
PWM_FREQ_HZ = 330


class PIDController:
    """단일 축 PID 컨트롤러.

    현재 설정에서 I 게인은 0이므로 I항과 적분 관련 상태는 제거했다.
    yaw/pitch는 같은 게인을 사용하지만 D항의 이전 오차 상태는 축별로
    독립적이어야 하므로 PIDController 인스턴스는 각각 유지한다.
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
############################################################self prev error 수정함, 차분 d 수정함
        if abs(error) < self.deadband:
            return 0.0

        p = self.kp * error
        d = -self.kd * velocity if use_d else 0.0

        output = p + d
        return max(-self.output_limit, min(self.output_limit, output))

    def reset(self):
        pass


class ServoController:
    """yaw/pitch 공통 PID 설정을 사용하는 2축 서보 컨트롤러.

    현재 서보 각도는 최대 각속도 기반의 단순 선형 모델로 추정한다.
    """

    def __init__(
        self,
        yaw_pin: int = YAW_PIN,
        pitch_pin: int = PITCH_PIN,
        kp: float = 1.1,
        kd: float = 0.011,
        dt: float = 1 / 30,
        output_limit: float = 4.0,
        deadband: float = 2.0,
        max_speed: float = 90.0,
        home_step_deg: float = 5.0,
        home_step_delay: float = 0.06,
        pwm_freq: float = PWM_FREQ_HZ,
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

        # 초기 각도 출력 전에 PWM 주파수/range를 먼저 설정해야 한다.
        self.pwm_freq = self._setup_pwm(pwm_freq)

        self.yaw_angle = 90.0
        self.pitch_angle = 90.0
        self.yaw_cmd_angle = 90.0
        self.pitch_cmd_angle = 90.0

        self.home_step_deg = home_step_deg
        self.home_step_delay = home_step_delay

        # 게인은 공통이지만 이전 오차 상태가 섞이지 않도록 인스턴스는 축별 유지.
        self.yaw_pid = PIDController(
            kp=kp,
            kd=kd,
            dt=dt,
            output_limit=output_limit,
            deadband=deadband,
        )
        self.pitch_pid = PIDController(
            kp=kp,
            kd=kd,
            dt=dt,
            output_limit=output_limit,
            deadband=deadband,
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

    def _setup_pwm(self, requested_hz: float) -> float:
        """서보 신호 주파수를 설정하고 실제 적용된 주파수를 반환한다.

        requested_hz <= 50: set_servo_pulsewidth() 사용 (pigpio 고정 50Hz).
        requested_hz  > 50: set_PWM_frequency + set_PWM_range + set_PWM_dutycycle.
            range를 주기(µs)로 맞춰 두면 dutycycle 값이 곧 펄스폭(µs)이 된다.
        """
        if requested_hz <= 50:
            self._use_servo_api = True
            return 50.0

        self._use_servo_api = False
        actual_hz = float(requested_hz)
        for pin in (self.yaw_pin, self.pitch_pin):
            # 가장 가까운 지원 주파수로 맞춰지며, 실제 적용된 값이 반환된다.
            actual_hz = float(self.pi.set_PWM_frequency(pin, int(round(requested_hz))))

        period_us = int(round(1e6 / actual_hz))
        for pin in (self.yaw_pin, self.pitch_pin):
            self.pi.set_PWM_range(pin, period_us)

        print(
            f"[ServoController] PWM {requested_hz:g}Hz 요청 → "
            f"{actual_hz:g}Hz 적용 (주기 {period_us}µs)"
        )
        if actual_hz > requested_hz * 1.1:
            print(
                "[ServoController] 경고: 적용 주파수가 요청값보다 높습니다. "
                "pigpiod 샘플레이트(-s 옵션)를 확인하세요 (기본 5µs 권장)."
            )
        return actual_hz

    def _set_pw(self, pin: int, pw: int):
        """펄스폭(µs)을 출력한다. pw=0이면 출력 정지."""
        if self._use_servo_api:
            self.pi.set_servo_pulsewidth(pin, pw)
        else:
            self.pi.set_PWM_dutycycle(pin, pw)

    @staticmethod
    def _step_toward(
        current: float,
        target: float,
        max_speed: float,
        dt: float,
    ) -> float:
        max_step = max_speed * dt
        diff = target - current
        if abs(diff) <= max_step:
            return target
        return current + math.copysign(max_step, diff)

    def _update_servo_position(self):
        self.yaw_angle = self._step_toward(
            self.yaw_angle,
            self.yaw_cmd_angle,
            self.max_speed,
            self.dt,
        )
        self.pitch_angle = self._step_toward(
            self.pitch_angle,
            self.pitch_cmd_angle,
            self.max_speed,
            self.dt,
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