import pigpio
import time

#################pin pin pin pin pin
YAW_PIN   = 17
PITCH_PIN = 27

PW_MIN  =  500
PW_MID  = 1500
PW_MAX  = 2500

YAW_MIN,   YAW_MAX   = 0, 180
PITCH_MIN, PITCH_MAX = 0, 180

DEADBAND = 2
STEP     = 1.0   # 고정 이동량

class ServoController:
    def __init__(self, yaw_pin=YAW_PIN, pitch_pin=PITCH_PIN):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("sudo pigpiod haseyo")

        self.yaw_pin   = yaw_pin
        self.pitch_pin = pitch_pin

        self.yaw_angle   = 90.0
        self.pitch_angle = 90.0

        self._set_pw(self.yaw_pin,   PW_MID)
        self._set_pw(self.pitch_pin, PW_MID)
        time.sleep(0.5)

    def _angle_to_pw(self, angle_deg):
        pw = PW_MID + (angle_deg / 180.0) * (PW_MAX - PW_MIN)
        return int(max(PW_MIN, min(PW_MAX, pw)))

    def _set_pw(self, pin, pw):
        self.pi.set_servo_pulsewidth(pin, pw)

    def set_yaw(self, angle_deg):
        angle_deg = max(YAW_MIN, min(YAW_MAX, angle_deg))
        self.yaw_angle = angle_deg
        self._set_pw(self.yaw_pin, self._angle_to_pw(angle_deg))

    def set_pitch(self, angle_deg):
        angle_deg = max(PITCH_MIN, min(PITCH_MAX, angle_deg))
        self.pitch_angle = angle_deg
        self._set_pw(self.pitch_pin, self._angle_to_pw(angle_deg))

    def move(self, yaw_deg, pitch_deg):
        # 데드밴드 벗어나면 고정 STEP만큼만 이동
        if abs(yaw_deg) >= DEADBAND:
            step = STEP if yaw_deg > 0 else -STEP
            self.set_yaw(self.yaw_angle + step)

        if abs(pitch_deg) >= DEADBAND:
            step = STEP if pitch_deg > 0 else -STEP
            self.set_pitch(self.pitch_angle + step)

    def center(self):
        self.set_yaw(90.0)
        self.set_pitch(90.0)

    def stop(self):
        self._set_pw(self.yaw_pin,   0)
        self._set_pw(self.pitch_pin, 0)
        self.pi.stop()