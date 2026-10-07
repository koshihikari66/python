import pigpio
import time

# ================= Configuration =================
PITCH_SERVO_PIN = 15    # GPIO pin number (BCM) - 나중에 수정
YAW_SERVO_PIN = 23      # GPIO pin number (BCM) - 나중에 수정

# Pitch: 90 ~ 180, initial 90
PITCH_MIN_ANGLE = 0
PITCH_MAX_ANGLE = 180
PITCH_INIT_ANGLE = 90

# Yaw: 0 ~ 180, initial 90
YAW_MIN_ANGLE = 0
YAW_MAX_ANGLE = 180
YAW_INIT_ANGLE = 90

MIN_PW = 500            # Pulse width for 0 degree (us)
MAX_PW = 2500           # Pulse width for 180 degree (us)

STEP_ANGLE = 5          # Angle step per update (deg)
UPDATE_DT = 0.3        # Time between updates (sec)
# =================================================

pi = pigpio.pi()

if not pi.connected:
    print("Failed to connect to pigpio daemon")
    exit()


def angle_to_pulsewidth(angle):
    """Convert angle (0~180 deg) to pulse width (us)."""
    return MIN_PW + (angle / 180.0) * (MAX_PW - MIN_PW)


def set_servo_angle(pin, angle):
    """Send pulse width to servo."""
    pw = angle_to_pulsewidth(angle)
    pi.set_servo_pulsewidth(pin, pw)


def move_one_step(current, target):
    """Move current angle one step toward target angle."""
    if abs(current - target) <= STEP_ANGLE:
        return target

    if current < target:
        return current + STEP_ANGLE
    else:
        return current - STEP_ANGLE


def move_pitch_yaw_smoothly(target_pitch, target_yaw):
    """Move pitch and yaw simultaneously toward target angles."""
    global current_pitch, current_yaw

    while (
        abs(current_pitch - target_pitch) > 0.001 or
        abs(current_yaw - target_yaw) > 0.001
    ):
        current_pitch = move_one_step(current_pitch, target_pitch)
        current_yaw = move_one_step(current_yaw, target_yaw)

        set_servo_angle(PITCH_SERVO_PIN, current_pitch)
        set_servo_angle(YAW_SERVO_PIN, current_yaw)

        time.sleep(UPDATE_DT)


# 현재 각도 초기값 설정
current_pitch = PITCH_INIT_ANGLE
current_yaw = YAW_INIT_ANGLE

try:
    print("Pitch-Yaw simultaneous control")
    print("Pitch range: 90 ~ 180")
    print("Yaw range: 0 ~ 180")
    print("Input format example: 120 45")
    print("Ctrl+C to exit")

    # 시작 시 기본 위치로 설정
    set_servo_angle(PITCH_SERVO_PIN, current_pitch)
    set_servo_angle(YAW_SERVO_PIN, current_yaw)

    while True:
        try:
            user_input = input("Target pitch yaw: ")
            target_pitch, target_yaw = map(float, user_input.split())

            if not (PITCH_MIN_ANGLE <= target_pitch <= PITCH_MAX_ANGLE):
                print("Pitch angle out of range")
                continue

            if not (YAW_MIN_ANGLE <= target_yaw <= YAW_MAX_ANGLE):
                print("Yaw angle out of range")
                continue

        except ValueError:
            print("Invalid input. Example: 120 45")
            continue

        # Pitch와 Yaw 동시 이동
        move_pitch_yaw_smoothly(target_pitch, target_yaw)

except KeyboardInterrupt:
    print("\nProgram terminated")

finally:
    print("Returning to initial position...")

    # 종료 전 기본 위치로 복귀
    move_pitch_yaw_smoothly(PITCH_INIT_ANGLE, YAW_INIT_ANGLE)

    # 기본 위치에서 잠깐 유지
    time.sleep(0.5)

    # 서보 신호 OFF
    pi.set_servo_pulsewidth(PITCH_SERVO_PIN, 0)
    pi.set_servo_pulsewidth(YAW_SERVO_PIN, 0)

    pi.stop()
    print("Servo stopped")
