import pigpio
import time
import math
import numpy as np

# ── 핀 설정 ────────────────────────────────────────────────
YAW_PIN   = 23
PITCH_PIN = 15

PW_MIN  =  500
PW_MID  = 1500
PW_MAX  = 2500

YAW_MIN,   YAW_MAX   = 0, 180
PITCH_MIN, PITCH_MAX = 90, 180


# ── PID 컨트롤러 ───────────────────────────────────────────
class PIDController:
    """
    단일 축 PID 컨트롤러.
    D항은 기본적으로 오차 차분 기반.
    velocity가 들어오면 기존 코드와 동일하게 -kd*velocity를 사용한다.
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

    def compute(self, error: float, velocity: float = 0.0, use_d: bool = True) -> float:
        """
        PID 출력 계산.

        Parameters
        ----------
        error    : 현재 오차 [deg]
        velocity : 칼만 필터 추정 각속도 [deg/s] 또는 외부 속도값
        use_d    : False면 D항 제거

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

        # D항
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

    핵심 수정:
    - 명령각(command angle)과 현재 추정각(actual estimated angle)을 분리했다.
    - self.yaw_angle / self.pitch_angle은 이제 '서보가 실제로 도달했다고 추정되는 각도'이다.
    - self.yaw_cmd_angle / self.pitch_cmd_angle은 PWM으로 보낸 '명령각'이다.
    - 현재 추정각은 선형(속도 제한) 모델로 명령각을 따라간다.

        diff = theta_cmd[k] - theta_actual[k]
        step = clamp(diff, -max_speed*dt, +max_speed*dt)
        theta_actual[k+1] = theta_actual[k] + step

      실제 RC서보 대부분이 "남은 거리에 비례해 감속"하는 1차 지연보다는
      "일정 각속도로 이동하다 도착 직전에만 멈추는" 동작에 가깝기 때문에
      1차 지연(지수) 모델 대신 이 선형 모델을 사용한다.
      max_speed[deg/s]는 서보 스펙(예: "90도 이동에 약 1초" → 약 90 deg/s)에서
      바로 대입하면 되므로 tau보다 튜닝이 직관적이다.

      use_imu=True로 설정하면 pitch_angle은 더 이상 이 선형 모델의 추정치가
      아니라 MPU6050 IMU(I2C)로 직접 측정한 실제 각도가 된다(mpu6050_imu.py).
      MPU6050은 BNO085와 달리 칩 내부 센서 퓨전이 없어서, mpu6050_imu.py가
      가속도계(드리프트 없음)+자이로(부드러운 응답)를 상보 필터로 합성해
      pitch를 추정한다.
      IMU 읽기가 실패한 프레임에서만 일시적으로 선형 모델로 대체(fallback)한다.
      yaw는 현재 90도 고정 축이라 계속 선형 모델을 사용한다.

    nnewredc.py에서 servo.yaw_angle, servo.pitch_angle을 읽고 있으므로,
    이 값이 즉시 명령각으로 점프하지 않고 실제 서보 지연을 반영하게 된다.

    축 파라미터 완전 분리:
    - PID 게인(kp/ki/kd), output_limit, integral_limit, deadband,
      cmd_smooth(EMA 스무딩), max_speed(최대 각속도)까지 전부
      yaw_* / pitch_* 접두사로 나눠서 독립적으로 튜닝 가능하다.
    - 두 축의 기구(서보 종류, 부하, 마찰 등) 특성이 다르면 이제
      각각 다른 값을 넣어 따로 조정하면 된다. (dt는 두 축이 같은
      제어 루프 주기를 쓰므로 공통 파라미터로 유지)
    """

    def __init__(
        self,
        yaw_pin: int   = YAW_PIN,
        pitch_pin: int = PITCH_PIN,

        # ── YAW 축 PID 게인 ────────────────────────────────
        yaw_kp: float = 1.1,
        yaw_ki: float = 0.0,
        yaw_kd: float = 0.011,

        # ── PITCH 축 PID 게인 ──────────────────────────────
        pitch_kp: float = 1.1,
        pitch_ki: float = 0.0,
        pitch_kd: float = 0.011,

        dt: float = 1 / 30,

        # 축별 출력/적분 제한, 데드밴드
        yaw_output_limit: float   = 4.0,
        pitch_output_limit: float = 4.0,

        yaw_integral_limit: float   = 30.0,
        pitch_integral_limit: float = 30.0,

        yaw_deadband: float   = 2.0,
        pitch_deadband: float = 2.0,

        # 서보 명령 EMA 스무딩 (0=즉시반응, 1=변화없음), 축별 분리
        # 1-3프레임 주기 검출 진동 억제용 (권장: 0.5~0.7)
        yaw_cmd_smooth: float   = 0.1,
        pitch_cmd_smooth: float = 0.1,

        # 서보 최대 각속도 [deg/s], 축별 분리.
        # 값이 클수록 실제각이 명령각을 더 빨리 따라감.
        # 예: "90도 이동에 약 1초" 스펙이면 대략 90.0 근처로 설정.
        yaw_max_speed: float   = 90.0,
        pitch_max_speed: float = 90.0,

        # 종료(stop()) 시 초기 위치(90/90)로 복귀할 때 사용하는 스텝 크기/간격.
        # 한 번에 점프하지 않고 이 각도(deg) 단위로 나눠서 서서히 이동한다.
        home_step_deg: float   = 5.0,
        home_step_delay: float = 0.06,

        # ── IMU(MPU6050) 기반 피치 실제각 측정 ──────────────
        # True로 켜면 _update_servo_position()에서 pitch_angle을
        # 선형(속도 제한) 모델 대신 MPU6050 실측값(가속도계+자이로 상보 필터)
        # 으로 갱신한다.
        # (yaw는 현재 90도 고정 축이라 IMU를 사용하지 않고 계속 선형 모델을 쓴다.)
        use_imu: bool = False,
        imu_i2c_addr: int = 0x68,
        imu_use_gyro_fusion: bool = True,
        imu_comp_alpha: float = 0.98,
        imu_pitch_sign: float = 1.0,
        imu_calib_samples: int = 30,
    ):
        self.pi = pigpio.pi()

        if not self.pi.connected:
            raise RuntimeError(
                "pigpiod가 실행 중이 아닙니다. "
                "'sudo pigpiod'를 먼저 실행하세요."
            )

        self.yaw_pin   = yaw_pin
        self.pitch_pin = pitch_pin

        self.dt = dt

        # 선형(속도 제한) 모델의 최대 각속도도 축별로 분리
        self.yaw_max_speed   = max(1e-6, yaw_max_speed)
        self.pitch_max_speed = max(1e-6, pitch_max_speed)

        # 현재 추정각: 외부 코드가 읽어야 하는 값
        # 즉, nnewredc.py의 servo.yaw_angle / servo.pitch_angle은 이 값을 사용한다.
        self.yaw_angle   = 90.0
        self.pitch_angle = 90.0

        # PWM으로 보낸 명령각
        self.yaw_cmd_angle   = 90.0
        self.pitch_cmd_angle = 90.0

        # 종료 시 초기 위치 복귀용 스텝 파라미터
        self.home_step_deg   = home_step_deg
        self.home_step_delay = home_step_delay

        # EMA 스무딩 상태 = 명령각 스무딩용 (축별 계수 분리)
        self.yaw_cmd_smooth   = yaw_cmd_smooth
        self.pitch_cmd_smooth = pitch_cmd_smooth
        self._smooth_yaw   = 90.0
        self._smooth_pitch = 90.0

        # YAW / PITCH 각각 독립된 PID 인스턴스 + 독립된 게인
        self.yaw_pid = PIDController(
            kp=yaw_kp,
            ki=yaw_ki,
            kd=yaw_kd,
            dt=dt,
            output_limit=yaw_output_limit,
            integral_limit=yaw_integral_limit,
            deadband=yaw_deadband,
        )
        self.pitch_pid = PIDController(
            kp=pitch_kp,
            ki=pitch_ki,
            kd=pitch_kd,
            dt=dt,
            output_limit=pitch_output_limit,
            integral_limit=pitch_integral_limit,
            deadband=pitch_deadband,
        )

        self._write_yaw_cmd(90.0)
        self._write_pitch_cmd(90.0)

        time.sleep(0.5)

        # ── IMU(MPU6050) 초기화 ──────────────────────────────
        # use_imu=True일 때만 시도한다. 여기서 실패하면(라이브러리 미설치,
        # I2C 배선/주소 문제 등) 예외를 그대로 올려서 문제를 바로 알 수 있게 한다
        # (하드웨어를 쓰기로 해놓고 조용히 선형 모델로 폴백하면 오히려 혼란스럽다).
        self.use_imu = use_imu
        self.imu = None
        if self.use_imu:
            from mpu6050_imu import MPU6050IMU

            self.imu = MPU6050IMU(
                i2c_addr=imu_i2c_addr,
                use_gyro_fusion=imu_use_gyro_fusion,
                comp_alpha=imu_comp_alpha,
                pitch_sign=imu_pitch_sign,
            )
            # 서보가 실제로 90도(수평 기준 자세)에 안정될 시간을 조금 더 준 뒤
            # 그 자세를 IMU의 0점으로 잡는다.
            time.sleep(0.3)
            self.imu.calibrate_zero(samples=imu_calib_samples)

    # ── 내부 헬퍼 ──────────────────────────────────────────
    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _angle_to_pw(self, angle_deg: float) -> int:
        pw = PW_MID + ((angle_deg - 90) / 180.0) * (PW_MAX - PW_MIN)
        return int(max(PW_MIN, min(PW_MAX, pw)))

    def _set_pw(self, pin: int, pw: int):
        self.pi.set_servo_pulsewidth(pin, pw)

    @staticmethod
    def _step_toward(current: float, target: float, max_speed: float, dt: float) -> float:
        """
        current에서 target을 향해 최대 각속도(max_speed)로 한 스텝 이동.
        남은 거리가 (max_speed*dt)보다 작으면 오버슈트 없이 target에 정확히 도달한다.
        """
        max_step = max_speed * dt
        diff = target - current
        if abs(diff) <= max_step:
            return target
        return current + math.copysign(max_step, diff)

    def _update_servo_position(self):
        """
        yaw_angle / pitch_angle('현재 서보모터 각도 추정/측정값')을 갱신한다.

        - yaw_angle   : 항상 선형(속도 제한) 모델로 추정한다(IMU 미사용 축).
        - pitch_angle : use_imu=True이면 MPU6050 실측값(상보 필터)을 사용한다.
                        IMU 읽기가 실패하면 그 프레임만 선형 모델로 대체한다.
        """
        self.yaw_angle = self._step_toward(
            self.yaw_angle, self.yaw_cmd_angle, self.yaw_max_speed, self.dt
        )

        if self.use_imu and self.imu is not None:
            try:
                self.pitch_angle = self._clamp(
                    self.imu.get_servo_pitch_angle(), PITCH_MIN, PITCH_MAX
                )
                return
            except Exception:
                # I2C 읽기 실패 등 - 이번 스텝만 선형 모델로 폴백
                pass

        self.pitch_angle = self._step_toward(
            self.pitch_angle, self.pitch_cmd_angle, self.pitch_max_speed, self.dt
        )

    # ── 명령각 설정: PWM 출력만 담당 ───────────────────────
    def _write_yaw_cmd(self, angle_deg: float):
        self.yaw_cmd_angle = self._clamp(angle_deg, YAW_MIN, YAW_MAX)
        self._set_pw(self.yaw_pin, self._angle_to_pw(self.yaw_cmd_angle))

    def _write_pitch_cmd(self, angle_deg: float):
        self.pitch_cmd_angle = self._clamp(angle_deg, PITCH_MIN, PITCH_MAX)
        self._set_pw(self.pitch_pin, self._angle_to_pw(self.pitch_cmd_angle))

    # 기존 코드와의 호환용.
    # 주의: set_yaw/set_pitch는 이제 '명령각'을 보내는 함수이고,
    # yaw_angle/pitch_angle은 _update_servo_position()에서 천천히 따라간다.
    def set_yaw(self, angle_deg: float):
        self._write_yaw_cmd(angle_deg)

    def set_pitch(self, angle_deg: float):
        self._write_pitch_cmd(angle_deg)

    # ── 메인 제어 ─────────────────────────────────────────
    def move(
        self,
        yaw_err: float,
        pitch_err: float,
        vx_kalman: float = 0.0,
        vy_kalman: float = 0.0,
        use_d: bool = True
    ):
        """
        Parameters
        ----------
        yaw_err   : 수평 오차 [deg]
        pitch_err : 수직 오차 [deg]
        vx_kalman : 칼만 추정 yaw 각속도 [deg/s]
        vy_kalman : 칼만 추정 pitch 각속도 [deg/s]
        use_d     : False면 D항 제거
        """
        yaw_cmd_delta = self.yaw_pid.compute(
            yaw_err, velocity=vx_kalman, use_d=use_d
        )
        pitch_cmd_delta = self.pitch_pid.compute(
            pitch_err, velocity=vy_kalman, use_d=use_d
        )

        # 현재 추정각 기준으로 다음 명령각 계산
        raw_yaw   = self.yaw_angle   - yaw_cmd_delta
        raw_pitch = self.pitch_angle + pitch_cmd_delta

        # EMA 스무딩: 급격한 방향 전환 억제
        # self._smooth_*는 명령각 스무딩 상태이다.
        # yaw_cmd_smooth / pitch_cmd_smooth로 축별 스무딩 세기를 독립적으로 조절.
        self._smooth_yaw = (
            (1 - self.yaw_cmd_smooth) * raw_yaw
            + self.yaw_cmd_smooth * self._smooth_yaw
        )
        self._smooth_pitch = (
            (1 - self.pitch_cmd_smooth) * raw_pitch
            + self.pitch_cmd_smooth * self._smooth_pitch
        )

        # 실제 PWM 명령 출력
        # 기존 코드처럼 yaw를 고정하려면 아래 2줄을 유지한다.
        #self.set_yaw(90.0)
        self.set_yaw(self._smooth_yaw)

        #self.set_pitch(90.0)
        self.set_pitch(self._smooth_pitch)

        # PWM 명령 후, 현재 서보 각도 추정값을 선형(속도 제한) 모델로 갱신
        self._update_servo_position()

    def _ramp_to(
        self,
        target_yaw: float,
        target_pitch: float,
        step_deg: float,
        step_delay: float,
    ):
        """
        현재 명령각(yaw_cmd_angle, pitch_cmd_angle)에서 목표각까지
        step_deg 단위로 나눠서 서서히 PWM 명령을 보낸다.

        Ctrl+C 등으로 갑자기 종료할 때 초기 위치(90/90)로 한 번에
        점프하지 않고, 5도 정도씩 나눠서 부드럽게 복귀시키는 용도.
        """
        start_yaw   = self.yaw_cmd_angle
        start_pitch = self.pitch_cmd_angle

        dist = max(abs(target_yaw - start_yaw), abs(target_pitch - start_pitch))
        if dist < 1e-6:
            return

        steps = max(1, math.ceil(dist / step_deg))

        for i in range(1, steps + 1):
            frac = i / steps
            self._write_yaw_cmd(start_yaw + (target_yaw - start_yaw) * frac)
            self._write_pitch_cmd(start_pitch + (target_pitch - start_pitch) * frac)
            time.sleep(step_delay)

    # ── 유틸리티 ──────────────────────────────────────────
    def center(self):
        self.yaw_pid.reset()
        self.pitch_pid.reset()

        self._smooth_yaw   = 90.0
        self._smooth_pitch = 90.0

        self._write_yaw_cmd(90.0)
        self._write_pitch_cmd(90.0)

        # center()는 기준점 재설정이므로 실제 추정각도 90도로 초기화
        self.yaw_angle   = 90.0
        self.pitch_angle = 90.0

    def recalibrate_imu(self):
        """
        서보가 실제로 명령각 90/90(수평 기준 자세)에 도달해 안정된 상태라고
        확신할 때 호출한다. 예: center() 호출 후 잠깐 대기했다가 호출.
        IMU를 쓰지 않을 때는 아무 동작도 하지 않는다.
        """
        if self.use_imu and self.imu is not None:
            self.imu.calibrate_zero()

    def stop(self):
        # 초기 위치(90/90)로 한 번에 점프하지 않고 home_step_deg(기본 5도)씩
        # 나눠서 서서히 복귀시킨다 (Ctrl+C 등 급정지 시 충격 방지).
        self._ramp_to(
            90.0, 90.0,
            step_deg=self.home_step_deg,
            step_delay=self.home_step_delay,
        )

        self.center()

        time.sleep(1)

        self._set_pw(YAW_PIN, 0)
        self._set_pw(PITCH_PIN, 0)

        self.pi.stop()

        if self.use_imu and self.imu is not None:
            self.imu.close()
