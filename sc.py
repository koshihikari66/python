import pigpio
import time

YAW_PIN   = 17
PITCH_PIN = 27

PW_MIN  =  500   # 0
PW_MID  = 1500   # 90
PW_MAX  = 2500   # 180

YAW_MIN,   YAW_MAX   = -90, 90
PITCH_MIN, PITCH_MAX = -90, 90

DEADBAND = 2
MAX_STEP = 5.0

class ServoController:
    def __init__(self, yaw_pin=YAW_PIN, pitch_pin=PITCH_PIN):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("sudo pigpiod haseyo")

        self.yaw_pin   = yaw_pin
        self.pitch_pin = pitch_pin

        # 현재 각도 상태
        self.yaw_angle   = 0.0
        self.pitch_angle = 0.0

        # 중립 위치로 초기화
        self._set_pw(self.yaw_pin,   PW_MID)
        self._set_pw(self.pitch_pin, PW_MID)
        time.sleep(0.5)

    # ── 내부 유틸 ────────────────────────────────────────────
    def _angle_to_pw(self, angle_deg):
        """각도(-90~90) → 펄스폭(us) 변환"""
        pw = PW_MID + (angle_deg / 90.0) * (PW_MAX - PW_MID)
        return int(max(PW_MIN, min(PW_MAX, pw)))

    def _set_pw(self, pin, pw):
        self.pi.set_servo_pulsewidth(pin, pw)

    # ── 퍼블릭 API ───────────────────────────────────────────
    def set_yaw(self, angle_deg):
        """Yaw 서보를 절대 각도로 이동"""
        angle_deg = max(YAW_MIN, min(YAW_MAX, angle_deg))
        self.yaw_angle = angle_deg
        self._set_pw(self.yaw_pin, self._angle_to_pw(angle_deg))

    def set_pitch(self, angle_deg):
        """Pitch 서보를 절대 각도로 이동"""
        angle_deg = max(PITCH_MIN, min(PITCH_MAX, angle_deg))
        self.pitch_angle = angle_deg
        self._set_pw(self.pitch_pin, self._angle_to_pw(angle_deg))

    def move(self, yaw_deg, pitch_deg):
        if abs(yaw_deg) >= DEADBAND:
            step = min(abs(yaw_deg), MAX_STEP) * (1 if yaw_deg > 0 else -1)
            self.set_yaw(self.yaw_angle + step)

        if abs(pitch_deg) >= DEADBAND:
            step = min(abs(pitch_deg), MAX_STEP) * (1 if pitch_deg > 0 else -1)
            self.set_pitch(self.pitch_angle + step)

    def center(self):
        """중립 위치로 복귀"""
        self.move(0, 0)

    def stop(self):
        """서보 신호 OFF 후 연결 해제"""
        self._set_pw(self.yaw_pin,   0)
        self._set_pw(self.pitch_pin, 0)
        self.pi.stop()

# ── 단독 실행 테스트 ─────────────────────────────────────────
if __name__ == "__main__":
    sc = ServoController()
    print("중립 → 좌 → 우 → 중립 테스트")
    try:
        sc.center()
        time.sleep(1)
        sc.move(-45, 20)
        time.sleep(1)
        sc.move( 45,-20)
        time.sleep(1)
        sc.center()
        time.sleep(1)
    finally:
        sc.stop()
        print("완료")