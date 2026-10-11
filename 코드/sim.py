"""
sim_stationary.py
────────────────────────────────────────────────────────────
정지해 있는 목표를 hybrid.py 의 제어 루프로 추적할 때 영상 중심 대비
오차(angle / yaw / pitch)가 시간에 따라 어떻게 변하는지 시뮬레이션한다.

hybrid.py 와 같은 구성
    측정(픽셀→각도) → 월드 각도 변환 → 등가속 칼만필터(LEDTrackerCA)
    → 축별 PID → 서보 각도 모델(split / tau)

hybrid.py 에는 없지만 "실제 장비에서 칼만필터가 영향을 주는 요소"로
아래 3가지를 옵션으로 넣었다. (아래 설정 구역에서 값 변경, 끄려면 0 / 1.0)
    1) 측정 노이즈       MEAS_NOISE_STD_DEG
    2) 카메라 지연       CAMERA_DELAY_FRAMES
    3) 서보 모델 오차    REAL_STEP_SCALE

실행하면 2가지만 묻는다.
    - tau 사용 여부 (y: 두 축 tau 방식 / n: 두 축 split 방식)
    - csv 저장 여부 (y: 저장 / n: 터미널 출력)
시뮬레이션 시간은 아래 SIM_TIME 을 직접 수정.

CSV 컬럼은 TalkFile_red2.xlsx 와 같다:  time_s, angle_deg, yaw_deg, pitch_deg
부호는 angle_logger 와 같다 (yaw: 오른쪽 +, pitch: 아래쪽 +).
"""

import csv
import math

import numpy as np

# ════════════════════════════════════════════════════════════
#  ★ 직접 수정하는 설정
# ════════════════════════════════════════════════════════════
SIM_TIME = 10.0          # 시뮬레이션 시간 [초]

# 시작 시점(서보 90/90)에서 목표가 영상 중심에서 벗어난 각도 [deg]
# yaw 오른쪽(+), pitch 아래쪽(+).  pitch 서보 범위가 90~180 이라
# 목표가 아래쪽(+)이면 hybrid 와 마찬가지로 따라가지 못한다.
INIT_YAW_DEG = 10.0
INIT_PITCH_DEG = -10.0

# ── 칼만필터가 영향을 주는 비이상적 요소 ─────────────────────
MEAS_NOISE_STD_DEG = 0.1   # 검출 위치 노이즈 표준편차 [deg]  (0.1° ≈ 0.8px). 0 → 노이즈 없음
CAMERA_DELAY_FRAMES = 1    # 영상이 찍힌 뒤 제어에 반영되기까지 지연 [프레임]. 0 → 지연 없음
REAL_STEP_SCALE = 0.8      # 실제 서보가 모델 예상 이동량의 몇 배만 움직이는지. 1.0 → 모델과 일치
RANDOM_SEED = 0            # 노이즈 재현용 시드. None 이면 매번 다름

# True  → 로그에 "실제(참) 오차" 기록
# False → hybrid 의 angle_logger 처럼 "측정값(노이즈·지연 포함)" 기록
LOG_TRUE_ERROR = False

SETTLE_BAND_DEG = 1.0      # 요약에서 "수렴"으로 볼 범위 [deg]
STEADY_WINDOW_S = 1.0      # 요약의 정상상태 RMS 를 구할 마지막 구간 [초]

# ════════════════════════════════════════════════════════════
#  hybrid.py / sccpid_hybrid.py 와 동일하게 맞춘 파라미터
#  (hybrid.py 값을 바꾸면 여기도 같이 바꿔줄 것)
# ════════════════════════════════════════════════════════════
CTRL_DT = 1 / 30

YAW_PID_KP = 0.13
YAW_PID_KD = 0.001
PITCH_KP_BASE = 0.13
PITCH_KD_BASE = 0.001
PID_OUTPUT_LIMIT = 3.0
PID_DEADBAND = 1.0

YAW_MAX_SPEED = 180.0     # deg/s
PITCH_MAX_SPEED = 90.0    # deg/s
TAU = 0.07                # tau 방식 시정수 [s]

YAW_MIN, YAW_MAX = 0.0, 180.0
PITCH_MIN, PITCH_MAX = 90.0, 180.0

REDETECT_RAMP_FRAMES = 2  # 첫 검출 직후 명령을 서서히 키우는 프레임 수

# LEDTrackerCA 생성 인자 (hybrid.py main() 과 동일)
KF_POS_NOISE = 5e-4
KF_VEL_NOISE = 0.8
KF_ACC_NOISE = 0.1
KF_MEAS_NOISE = 0.3


# ════════════════════════════════════════════════════════════
#  등가속 칼만필터 (hybrid.LEDTrackerCA 와 같은 모델, blend_alpha=0)
#  상태 [yaw_w, pitch_w, ω_yaw, ω_pitch, α_yaw, α_pitch] / 측정 [yaw_w, pitch_w]
# ════════════════════════════════════════════════════════════
class KalmanCA:
    def __init__(self, dt=CTRL_DT):
        dt2 = 0.5 * dt ** 2
        self.F = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)
        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.Q = np.diag([KF_POS_NOISE] * 2 + [KF_VEL_NOISE] * 2 + [KF_ACC_NOISE] * 2)
        self.R = np.eye(2) * KF_MEAS_NOISE
        self.P = np.eye(6)
        self.x = np.zeros((6, 1))
        self.initialized = False

    def update(self, yaw_w, pitch_w):
        z = np.array([[yaw_w], [pitch_w]])
        if not self.initialized:
            self.x = np.array([[yaw_w], [pitch_w], [0.0], [0.0], [0.0], [0.0]])
            self.initialized = True

        # predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # correct
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P

        return self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]


# ════════════════════════════════════════════════════════════
#  PID  (sccpid_hybrid.PIDController 와 동일 로직)
# ════════════════════════════════════════════════════════════
class PID:
    def __init__(self, kp, kd, dt, output_limit, deadband):
        self.kp = kp
        self.kd = kd
        self.dt = dt
        self.output_limit = output_limit
        self.deadband = deadband
        self._prev_error = 0.0

    def compute(self, error, velocity=0.0):
        if abs(error) < self.deadband:
            error = 0.0

        p = self.kp * error
        if abs(velocity) > 1e-6:
            d = -self.kd * velocity
        else:
            d = self.kd * (error - self._prev_error) / self.dt

        self._prev_error = error
        out = p + d
        return max(-self.output_limit, min(self.output_limit, out))


# ════════════════════════════════════════════════════════════
#  1축: PID + 서보 모델(제어기가 아는 각도) + 실제 서보(참 각도)
# ════════════════════════════════════════════════════════════
class Axis:
    """
    sign     : 서보 각도를 움직이는 방향 (yaw: -1, pitch: +1)
               hybrid.py 의  yaw_angle - delta / pitch_angle + delta  와 같다.
    s_model  : 제어기가 믿는 서보 각도 (servo.yaw_angle / pitch_angle)
    s_real   : 실제 서보 각도 — 영상에 보이는 목표 위치는 이것으로 결정된다.
               REAL_STEP_SCALE=1.0 이면 s_model 과 항상 같다.
    use_tau  : True  → pitch 방식 (1차 지연 + 속도제한, 게인/alpha 보정)
               False → yaw 방식   (속도제한만)
    """

    def __init__(self, sign, rel_err0, lo, hi, max_speed,
                 kp_base, kd_base, use_tau, dt=CTRL_DT):
        self.sign = sign
        self.lo, self.hi = lo, hi
        self.use_tau = use_tau
        self.max_step = max_speed * dt
        self.alpha = 1.0 - math.exp(-dt / TAU)

        scale = 1.0 / self.alpha if use_tau else 1.0
        self.pid = PID(kp_base * scale, kd_base * scale, dt,
                       PID_OUTPUT_LIMIT, PID_DEADBAND)

        self.s_model = 90.0
        self.s_real = 90.0
        self.s_real_hist = []
        # rel_error = sign * (s_target - s) 가 시작 시 rel_err0 이 되도록 설정
        self.s_target = 90.0 + sign * rel_err0

    def true_rel(self, s):
        """서보가 s 일 때 영상 중심 대비 목표 각도 (hybrid 부호: 오른쪽+/위+)."""
        return self.sign * (self.s_target - s)

    def measured_rel(self, k, delay, noise):
        """k 프레임에서 제어기가 받는 측정값: delay 프레임 전 영상 + 노이즈."""
        s = self.s_real_hist[max(0, k - delay)]
        return self.true_rel(s) + noise

    def _advance(self, angle, cmd, scale=1.0):
        if self.use_tau:
            step = (cmd - angle) * self.alpha
            step = max(-self.max_step, min(self.max_step, step))
        else:
            diff = cmd - angle
            step = diff if abs(diff) <= self.max_step else math.copysign(self.max_step, diff)
        return angle + step * scale

    def move(self, err, velocity):
        delta = self.pid.compute(err, velocity)
        cmd = max(self.lo, min(self.hi, self.s_model + self.sign * delta))
        self.s_model = self._advance(self.s_model, cmd)
        self.s_real = self._advance(self.s_real, cmd, REAL_STEP_SCALE)


# ════════════════════════════════════════════════════════════
#  시뮬레이션
# ════════════════════════════════════════════════════════════
def _to_log_angles(yaw_rel, pitch_rel):
    """hybrid 부호(오른쪽+/위+) → angle_logger 부호(오른쪽+/아래+) 및 전체각도."""
    yaw_deg = yaw_rel
    pitch_deg = -pitch_rel
    xn = math.tan(math.radians(yaw_deg))
    yn = math.tan(math.radians(pitch_deg))
    angle_deg = math.degrees(math.atan(math.hypot(xn, yn)))
    return angle_deg, yaw_deg, pitch_deg


def simulate(use_tau, sim_time=None):
    """[(time_s, angle_deg, yaw_deg, pitch_deg), ...] 반환."""
    sim_time = SIM_TIME if sim_time is None else sim_time
    rng = np.random.default_rng(RANDOM_SEED)

    yaw = Axis(-1, INIT_YAW_DEG, YAW_MIN, YAW_MAX, YAW_MAX_SPEED,
               YAW_PID_KP, YAW_PID_KD, use_tau)
    # hybrid 의 pitch 는 위쪽이 +. 로그(아래쪽 +) 부호의 반대.
    pitch = Axis(+1, -INIT_PITCH_DEG, PITCH_MIN, PITCH_MAX, PITCH_MAX_SPEED,
                 PITCH_KP_BASE, PITCH_KD_BASE, use_tau)
    kf = KalmanCA()

    n_steps = int(round(sim_time / CTRL_DT))
    rows = []
    for k in range(n_steps + 1):
        ramp = min(k + 1, REDETECT_RAMP_FRAMES) / REDETECT_RAMP_FRAMES

        # ① 이번 프레임 시작 시점의 실제 서보 각도 기록 (지연 처리용)
        yaw.s_real_hist.append(yaw.s_real)
        pitch.s_real_hist.append(pitch.s_real)

        # ② 측정: 지연된 영상 + 노이즈  → 상대각도
        n_yaw, n_pitch = rng.normal(0.0, MEAS_NOISE_STD_DEG, 2) if MEAS_NOISE_STD_DEG > 0 else (0.0, 0.0)
        yaw_meas = yaw.measured_rel(k, CAMERA_DELAY_FRAMES, n_yaw)
        pitch_meas = pitch.measured_rel(k, CAMERA_DELAY_FRAMES, n_pitch)

        # ③ 월드 각도로 변환 후 칼만필터 (hybrid.py 와 동일: 모델 각도 사용)
        yaw_world = -yaw.s_model + yaw_meas
        pitch_world = pitch.s_model + pitch_meas
        pred_yaw_w, pred_pitch_w, w_yaw, w_pitch = kf.update(yaw_world, pitch_world)

        # ④ 제어 오차 → PID → 서보 이동
        yaw_err = pred_yaw_w + yaw.s_model
        pitch_err = pred_pitch_w - pitch.s_model

        # ⑤ 기록 (이동 전 값, hybrid 의 log_pixel 위치와 동일)
        if LOG_TRUE_ERROR:
            y_rel = yaw.true_rel(yaw.s_real)
            p_rel = pitch.true_rel(pitch.s_real)
        else:
            y_rel, p_rel = yaw_meas, pitch_meas
        rows.append((k * CTRL_DT, *_to_log_angles(y_rel, p_rel)))

        yaw.move(yaw_err * ramp, w_yaw)
        pitch.move(pitch_err * ramp, w_pitch)

    return rows


# ════════════════════════════════════════════════════════════
#  입력 / 출력
# ════════════════════════════════════════════════════════════
def ask_yes_no(prompt):
    yes = {"y", "yes", "1", "o", "ㅇ", "예", "네", "사용", "tau", "저장", "true"}
    no = {"n", "no", "0", "x", "ㄴ", "아니오", "아니요", "미사용", "split", "false"}
    while True:
        s = input(prompt).strip().lower()
        if s in yes:
            return True
        if s in no:
            return False
        print("   y 또는 n 으로 입력하세요.")


def print_summary(rows, use_tau):
    n_win = max(1, int(round(STEADY_WINDOW_S / CTRL_DT)))
    print(f"\n[요약] {'tau' if use_tau else 'split'} | {SIM_TIME:g}s | "
          f"시작 yaw={INIT_YAW_DEG:+.1f}°, pitch={INIT_PITCH_DEG:+.1f}° | "
          f"노이즈 {MEAS_NOISE_STD_DEG:g}°, 지연 {CAMERA_DELAY_FRAMES}f, 서보배율 {REAL_STEP_SCALE:g}")
    print(f"  {'값':<10}{'최대|값|':>10}{'정상상태 RMS':>14}{f'≤{SETTLE_BAND_DEG:g}° 도달':>14}")
    for idx, name in ((1, "angle_deg"), (2, "yaw_deg"), (3, "pitch_deg")):
        vals = [r[idx] for r in rows]
        peak = max(abs(v) for v in vals)
        tail = vals[-n_win:]
        rms = math.sqrt(sum(v * v for v in tail) / len(tail))
        settle = next((r[0] for r in rows if abs(r[idx]) <= SETTLE_BAND_DEG), None)
        settle_s = f"{settle:.3f} s" if settle is not None else "도달 못 함"
        print(f"  {name:<10}{peak:>9.3f}°{rms:>13.3f}°{settle_s:>14}")


def main():
    use_tau = ask_yes_no("tau 사용? (y: tau 방식 / n: split 방식): ")
    save_csv = ask_yes_no("csv로 저장? (y: 저장 / n: 터미널 출력): ")

    rows = simulate(use_tau)

    if save_csv:
        path = f"sim_{'tau' if use_tau else 'split'}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["time_s", "angle_deg", "yaw_deg", "pitch_deg"])
            for t, a, y, p in rows:
                w.writerow([f"{t:.4f}", f"{a:.4f}", f"{y:.4f}", f"{p:.4f}"])
        print(f"\n저장 완료: {path}  ({len(rows)}행)")
    else:
        print("\ntime_s,angle_deg,yaw_deg,pitch_deg")
        for t, a, y, p in rows:
            print(f"{t:.4f},{a:.4f},{y:.4f},{p:.4f}")

    print_summary(rows, use_tau)


if __name__ == "__main__":
    main()