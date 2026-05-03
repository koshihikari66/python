import pigpio
import time

#px, py, vx, vy, ax, ay = prediction
#yaw_err, pitch_err = xy2angle.pixel_to_angles(px, py)
#servo.move(yaw_err, pitch_err, vx_kalman=vx, vy_kalman=vy)
 
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
 
    Parameters
    ----------
    kp, ki, kd   : PID 게인
    dt           : 제어 주기 (초). red.py 루프 주기와 맞춰주세요.
    output_limit : 한 스텝당 최대 각도 변화량 [deg] (포화 방지)
    integral_limit: 적분 와인드업 방지 한계값
    deadband     : 이 범위 내의 오차는 0으로 처리 (채터링 억제)
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
 
        self._integral  = 0.0
        self._prev_error = 0.0
 
    def compute(self, error: float, velocity: float = 0.0) -> float:
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
        if velocity != 0.0:
            d = -self.kd * velocity          # 속도 방향과 반대로 감쇠
        else:
            d = self.kd * (error - self._prev_error) / self.dt
 
        self._prev_error = error
 
        output = p + i + d
        return max(-self.output_limit, min(self.output_limit, output))
 
    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
 
 
# ── 서보 컨트롤러 (PID) ────────────────────────────────────
class ServoController:
    """
    YAW / PITCH 독립 PID 제어 서보 컨트롤러.
 
    move(yaw_err, pitch_err) 호출 시 오차 [deg] 를 받아
    PID 출력만큼 서보를 움직입니다.
 
    칼만 필터의 속도 추정값(vx, vy)을 선택적으로 전달하면
    D항을 칼만 속도로 대체하여 더 부드러운 제어가 됩니다.
 
    사용 예 (red.py):
        prediction = tracker.update(cx, cy)
        px, py, vx, vy, ax, ay = prediction
        yaw_err, pitch_err = xy2angle.pixel_to_angles(px, py)
        servo.move(yaw_err, pitch_err, vx_kalman=vx, vy_kalman=vy)
    """
 
    def __init__(
        self,
        yaw_pin: int   = YAW_PIN,
        pitch_pin: int = PITCH_PIN,
        # PID 게인 — 실물 튜닝 시 여기서 조정
        kp: float = 0.23,
        ki: float = 0.005,
        kd: float = 0.01,
        dt: float = 1 / 30,
        output_limit: float   = 5.0,
        integral_limit: float = 30.0,
        deadband: float       = 0.5,
    ):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpiod가 실행 중이 아닙니다. 'sudo pigpiod'를 먼저 실행하세요.")
 
        self.yaw_pin   = yaw_pin
        self.pitch_pin = pitch_pin
 
        self.yaw_angle   = 90.0
        self.pitch_angle = 90.0
 
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
 
        #print(
        #    f"[ServoController] 초기화 완료 | "
        #    f"Kp={kp}  Ki={ki}  Kd={kd}  dt={dt:.4f}s  "
        #    f"limit=±{output_limit}°  deadband=±{deadband}°"
        #)
 
    # ── 내부 헬퍼 ──────────────────────────────────────────
    def _angle_to_pw(self, angle_deg: float) -> int:
        pw = PW_MID + ((angle_deg - 90) / 180.0) * ((PW_MAX - PW_MIN))
        return int(max(PW_MIN, min(PW_MAX, pw)))
 
    def _set_pw(self, pin: int, pw: int):
        self.pi.set_servo_pulsewidth(pin, pw)
 
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
        yaw_err   : 수평 오차 [deg]  — xy2angle 출력값 그대로
        pitch_err : 수직 오차 [deg]  — xy2angle 출력값 그대로
        vx_kalman : 칼만 추정 x 속도 [px/frame] (선택)
        vy_kalman : 칼만 추정 y 속도 [px/frame] (선택)
        """
        yaw_cmd   = self.yaw_pid.compute(yaw_err,   velocity=vx_kalman)
        pitch_cmd = self.pitch_pid.compute(pitch_err, velocity=vy_kalman)
 
        self.set_yaw(self.yaw_angle - yaw_cmd)
        self.set_pitch(self.pitch_angle + pitch_cmd)
 
    # ── 유틸리티 ───────────────────────────────────────────
    def center(self):
        """서보를 중앙(90°)으로 복귀하고 PID 상태 초기화."""
        self.yaw_pid.reset()
        self.pitch_pid.reset()
        self.set_yaw(90.0)
        self.set_pitch(90.0)
 
    def stop(self):
        """PWM 신호 정지 및 pigpio 연결 해제."""
        #self._set_pw(self.yaw_pin,   0)
        #self._set_pw(self.pitch_pin, 0)
        self.set_yaw(90.0)
        self.set_pitch(90.0)
        self.pi.stop()
