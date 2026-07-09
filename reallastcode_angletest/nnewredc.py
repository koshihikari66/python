import cv2
import numpy as np
import threading
import xy2angle
from angle_logger import AngleLogger, combined_angle
# sudo fuser -k /dev/video2
from sccpid_first_order import ServoController

# ── YAW / PITCH 서보 파라미터 (축별 완전 분리) ──────────────
# 두 축의 서보/부하 특성이 다를 수 있으므로 PID 게인, 출력/적분 제한,
# 데드밴드, EMA 스무딩, 1차 지연 시정수까지 전부 독립적으로 튜닝한다.
# (현재 yaw는 move()에서 고정 90도로 잠겨 있어 실제 구동되지 않지만,
#  나중에 yaw 구동을 켜더라도 바로 사용할 수 있도록 값을 미리 분리해둔다.)
YAW_KP,   YAW_KI,   YAW_KD   = 0.12, 0.0, 0.002
PITCH_KP, PITCH_KI, PITCH_KD = 0.12, 0.0, 0.002

YAW_OUTPUT_LIMIT   = 1.0
PITCH_OUTPUT_LIMIT = 1.0

YAW_INTEGRAL_LIMIT   = 30.0
PITCH_INTEGRAL_LIMIT = 30.0

YAW_DEADBAND   = 1
PITCH_DEADBAND = 1

# 서보 명령 EMA 스무딩 (0=즉시반응, 1=변화없음). 축별 독립 조절.
YAW_CMD_SMOOTH   = 0.1
PITCH_CMD_SMOOTH = 0.1

# 서보 최대 각속도 [deg/s]. 축별 서보 기구 특성이 다르면 다르게 설정.
# 예: "90도 이동에 약 1초" 스펙이면 대략 90.0 근처.
YAW_MAX_SPEED   = 80.0
PITCH_MAX_SPEED = 30.0

# 종료(Ctrl+C) 시 초기 위치(90/90)로 복귀할 때, 한 번에 점프하지 않고
# 이 각도(deg) 단위로 나눠서 서서히 이동한다. step_delay는 스텝 사이 대기 시간[s].
HOME_STEP_DEG   = 6.0
HOME_STEP_DELAY = 0.1

servo = ServoController(
    yaw_kp=YAW_KP, yaw_ki=YAW_KI, yaw_kd=YAW_KD,
    pitch_kp=PITCH_KP, pitch_ki=PITCH_KI, pitch_kd=PITCH_KD,
    yaw_output_limit=YAW_OUTPUT_LIMIT, pitch_output_limit=PITCH_OUTPUT_LIMIT,
    yaw_integral_limit=YAW_INTEGRAL_LIMIT, pitch_integral_limit=PITCH_INTEGRAL_LIMIT,
    yaw_deadband=YAW_DEADBAND, pitch_deadband=PITCH_DEADBAND,
    yaw_cmd_smooth=YAW_CMD_SMOOTH, pitch_cmd_smooth=PITCH_CMD_SMOOTH,
    yaw_max_speed=YAW_MAX_SPEED, pitch_max_speed=PITCH_MAX_SPEED,
    home_step_deg=HOME_STEP_DEG, home_step_delay=HOME_STEP_DELAY,
)

from flask import Flask, Response

app = Flask(__name__)

# 최신 프레임을 스레드 간 공유하는 버퍼
# Condition: 새 프레임이 올 때까지 제너레이터를 블로킹 대기시킴
_frame_cond  = threading.Condition()
_latest_jpeg = None   # bytes | None


def _encode_jpeg(frame: np.ndarray, quality: int = 60) -> bytes:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _mjpeg_generator():
    """Flask 스트리밍 제너레이터 — 새 프레임이 도착할 때만 전송."""
    while True:
        with _frame_cond:
            _frame_cond.wait()
            jpeg = _latest_jpeg
        if jpeg is None:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
        )


@app.route('/video')
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/')
def index():
    return '<img src="/video" style="max-width:100%">'

# ── 파라미터 ──────────────────────────────────────────────
CAM_ID       = 0
WIDTH        = 640
HEIGHT       = 480

H_LOW1,  S_LOW1,  V_LOW1  =   0, 140, 140
H_HIGH1, S_HIGH1, V_HIGH1 =  12, 255, 255
H_LOW2,  S_LOW2,  V_LOW2  = 168, 140, 140
H_HIGH2, S_HIGH2, V_HIGH2 = 180, 255, 255

MIN_AREA   = 1
MAX_AREA   = 500

# 0.0 : 순수 correction (현재 프레임 최적 추정)
# 1.0 : 순수 prediction (다음 프레임 예측)
# 카메라가 움직이는 환경에서는 0.3~0.5 권장
BLEND_ALPHA = 0.9   # 0.0=correction만, 1.0=prediction만 (반드시 0~1)

# 몇 프레임 앞을 예측할지 — 이 값으로 레이턴시 보상량 조절
# 서보 지연 ≈ N프레임이면 N으로 설정 (권장: 2~5)
N_PREDICT   = 2

# 감속 감지 시 blend_alpha를 줄이는 민감도
# 값이 클수록 작은 가속도에도 blend를 강하게 줄임 (권장: 0.3~0.8)
DECEL_SENS  = 0

# predict_only → 재검출 복귀 시 서보 명령을 서서히 키우는 프레임 수
# 재검출 직후 큰 보정이 튀는 것을 방지 (권장: 2~4)
REDETECT_RAMP_FRAMES = 2

# ── Optical Flow 파라미터 ─────────────────────────────────
# Lucas-Kanade sparse OF: HSV 검출 실패 시 색상 강조 채널로 LED 추적
# 입력 채널: 2R-G-B (빨간색 강조, 흰색/파란색 배경 억제)
LK_PARAMS = dict(
    winSize  = (11, 15),
    maxLevel = 2,
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03,
    ),
)
# OF 결과를 칼만에 넘길 때 측정 노이즈 배율
# 1.0=HSV와 동일 신뢰, 2.0~4.0=HSV보다 덜 신뢰
OF_MEAS_NOISE_SCALE = 3.0
# OF 추적 위치의 2R-G-B 평균값이 이 값 미만이면 드리프트로 판단하고 무효화
OF_MIN_RED  = 14
# Forward-Backward 허용 오차 [픽셀]. 이 값 초과 시 추적 신뢰 불가
FB_MAX_ERR  = 4.0


# ── 등가속 칼만 필터 (월드 각도 공간) ────────────────────
class LEDTrackerCA:
    """
    상태 벡터: [yaw_world, pitch_world, ω_yaw, ω_pitch, α_yaw, α_pitch]
    측정 벡터: [yaw_world, pitch_world]

    픽셀이 아닌 월드 각도 공간에서 추적하므로,
    카메라(서보)가 움직여도 정지한 LED의 상태값이 변하지 않습니다.

    월드 각도 = 현재 서보 각도 + 픽셀→상대각도(xy2angle)

    수정 이력:
    - 월드 각도 공간으로 전환 (픽셀 기반 → 각도 기반)
    - dt=1/30 으로 실제 초 단위 사용
    - _was_missing 플래그로 복귀 첫 프레임 이중예측(double prediction) 버그 수정
    - blend_alpha=0 시 예측 루프 생략 (KF 상태 오염 방지)
    - update() 내 감속 감지 시 blend_alpha 동적 축소 (오버슈트 억제)
    - predict_only()에도 blend 적용하여 update()↔predict_only() 전환 시 타겟 위치 점프 제거
    """

    def __init__(self, dt: float = 1/30,
                 pos_noise: float = 1e-2,
                 vel_noise: float = 5,
                 acc_noise: float = 0.5,
                 meas_noise: float = 0.05,
                 max_missing: int = 5,
                 blend_alpha: float = BLEND_ALPHA,
                 n_predict: int = N_PREDICT,
                 decel_sens: float = DECEL_SENS):
        self.kf = cv2.KalmanFilter(6, 2)
        self.initialized  = False
        self.dt           = dt
        self.max_missing  = max_missing
        self.miss_count   = 0
        self.blend_alpha  = blend_alpha
        self.n_predict    = n_predict
        self.decel_sens   = decel_sens
        self._was_missing = False   # 복귀 첫 프레임 이중예측 방지 플래그

        dt2 = 0.5 * dt ** 2
        self.kf.transitionMatrix = np.array([
            [1, 0, dt,  0, dt2,   0],
            [0, 1,  0, dt,   0, dt2],
            [0, 0,  1,  0,  dt,   0],
            [0, 0,  0,  1,   0,  dt],
            [0, 0,  0,  0,   1,   0],
            [0, 0,  0,  0,   0,   1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.zeros((2, 6), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0
        self.kf.measurementMatrix[1, 1] = 1.0

        self.kf.processNoiseCov = np.diag([
            pos_noise, pos_noise,
            vel_noise, vel_noise,
            acc_noise, acc_noise,
        ]).astype(np.float32)

        self.kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * meas_noise
        )

        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

    def update(self, yaw_world: float, pitch_world: float):
        """
        월드 각도 측정값으로 predict → correct 처리.

        복귀 첫 프레임(_was_missing=True)에서는 predict()를 생략한다.
        predict_only()에서 이미 statePost가 최신 예측값이므로
        여기서 한 번 더 predict()하면 이중예측이 발생하기 때문.

        [Fix 1] 감속 감지 시 blend_alpha 동적 축소:
          corrected 상태의 가속도(α) 크기에 따라 blend_alpha를 줄여
          멈출 때 오버슈트를 억제한다.

        반환: (pred_yaw, pred_pitch, ω_yaw, ω_pitch, α_yaw, α_pitch)
          - pred_yaw/pitch : correction과 next_prediction을 α로 보간한 월드 각도
          - ω_yaw, ω_pitch : correction 기준 각속도 [deg/s] (PID D항에 사용)
        """
        measurement = np.array([[yaw_world], [pitch_world]], dtype=np.float32)

        if not self.initialized:
            self.kf.statePost = np.array(
                [[yaw_world], [pitch_world], [0.], [0.], [0.], [0.]],
                dtype=np.float32
            )
            self.initialized  = True
            self._was_missing = False

        self.miss_count = 0

        # 복귀 첫 프레임: statePost가 이미 최신 예측값 → predict 생략
        # 정상 프레임: predict → correct 순서
        if not self._was_missing:
            self.kf.predict()
        self._was_missing = False

        corrected = self.kf.correct(measurement)

        # ── [Fix 1] 감속 감지 → 동적 blend_alpha ────────────────
        # 가속도 크기가 클수록 (강하게 감속/가속 중) blend 비율을 줄여
        # 오버슈트를 억제한다.
        alpha_yaw   = corrected[4, 0]
        alpha_pitch = corrected[5, 0]
        decel_mag   = abs(alpha_yaw) + abs(alpha_pitch)
        dynamic_alpha = self.blend_alpha * max(0.0, 1.0 - self.decel_sens * decel_mag)

        # ── blend용: n_predict 프레임 앞 예측 후 상태 복원 ───────
        if dynamic_alpha > 0:
            state_snap     = self.kf.statePost.copy()
            cov_snap       = self.kf.errorCovPost.copy()
            state_pre_snap = self.kf.statePre.copy()
            cov_pre_snap   = self.kf.errorCovPre.copy()

            next_pred = corrected
            for _ in range(self.n_predict):
                next_pred = self.kf.predict()

            self.kf.statePost    = state_snap
            self.kf.errorCovPost = cov_snap
            self.kf.statePre     = state_pre_snap
            self.kf.errorCovPre  = cov_pre_snap

            byaw   = (1 - dynamic_alpha) * corrected[0, 0] + dynamic_alpha * next_pred[0, 0]
            bpitch = (1 - dynamic_alpha) * corrected[1, 0] + dynamic_alpha * next_pred[1, 0]
        else:
            byaw   = corrected[0, 0]
            bpitch = corrected[1, 0]

        return (
            byaw, bpitch,
            corrected[2, 0],   # ω_yaw   — correction 기준 각속도 [deg/s]
            corrected[3, 0],   # ω_pitch — correction 기준 각속도 [deg/s]
            corrected[4, 0],   # α_yaw
            corrected[5, 0],   # α_pitch
        )

    def predict_only(self):
        """
        검출 실패 시 호출. 예측만 수행하고 miss_count 증가.
        max_missing 초과 시 트래커를 초기화하고 None 반환.

        [Fix 2] update()와 동일한 blend 로직을 적용하여
        update()↔predict_only() 전환 시 타겟 위치 점프를 제거.

        statePost = statePre 로 복사하여 연속 miss 시 체이닝이 올바르게 동작.
        _was_missing 플래그를 True로 설정하여 복귀 첫 프레임의
        이중예측을 방지한다.
        """
        self.miss_count += 1
        #print(self.miss_count)
        if self.miss_count > self.max_missing:
            self.reset()
            return None

        predicted = self.kf.predict()
        self.kf.statePost    = self.kf.statePre.copy()   # 연속 miss 체이닝용
        self.kf.errorCovPost = self.kf.errorCovPre.copy()
        self._was_missing    = True

        # ── [Fix 2] predict_only에도 blend 적용 ─────────────────
        # update()와 동일한 방식으로 n_predict-1 스텝을 추가 예측.
        # (이미 1번 predict 했으므로 n_predict-1번 더 진행)
        if self.blend_alpha > 0 and self.n_predict > 1:
            state_snap = self.kf.statePost.copy()
            cov_snap   = self.kf.errorCovPost.copy()

            next_pred = predicted
            for _ in range(self.n_predict - 1):
                next_pred = self.kf.predict()

            # 상태 복원
            self.kf.statePost    = state_snap
            self.kf.errorCovPost = cov_snap
            self.kf.statePre     = self.kf.statePre   # 유지

            byaw   = (1 - self.blend_alpha) * predicted[0, 0] + self.blend_alpha * next_pred[0, 0]
            bpitch = (1 - self.blend_alpha) * predicted[1, 0] + self.blend_alpha * next_pred[1, 0]
            return (byaw, bpitch,
                    predicted[2, 0], predicted[3, 0],
                    predicted[4, 0], predicted[5, 0])

        return self._unpack(predicted)

    @staticmethod
    def _unpack(state):
        return (state[0, 0], state[1, 0],
                state[2, 0], state[3, 0],
                state[4, 0], state[5, 0])

    def reset(self):
        self.initialized  = False
        self.miss_count   = 0
        self._was_missing = False


def detect_red_led(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv,
                        (H_LOW1, S_LOW1, V_LOW1),
                        (H_HIGH1, S_HIGH1, V_HIGH1))
    mask2 = cv2.inRange(hsv,
                        (H_LOW2, S_LOW2, V_LOW2),
                        (H_HIGH2, S_HIGH2, V_HIGH2))
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [], mask

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        x, y, w, h = cv2.boundingRect(c)

        results.append({
            'centroid': (cx, cy),
            'area': area,
            'bbox': (x, y, w, h)
        })

    return results, mask


def draw_results(frame: np.ndarray, prediction: tuple, of_active: bool = False):
    """
    화면 중심(주점)에 조준 마커 + ±5도 정사각형을 항상 표시.
    - HSV 검출: 흰색 마커 + 'HSV' (녹색)
    - OF 추적:  노란색 마커 + 'OF'  (노란색)
    - 추적 없음: 회색 마커 + 'LOST' (회색)
    """
    vis = frame.copy()

    pcx = int(xy2angle.getcx())
    pcy = int(xy2angle.getcy())

    # ── ±5도 정사각형 (항상 표시) ─────────────────────────
    deg4 = np.radians(5.0)
    dx   = int(round(xy2angle.getfx() * np.tan(deg4)))
    dy   = dx
    cv2.rectangle(vis,
                  (pcx - dx, pcy - dy),
                  (pcx + dx, pcy + dy),
                  (0, 220, 220), 1, cv2.LINE_AA)

    # ── 조준 마커 & 상태 텍스트 ───────────────────────────
    if prediction is None:
        color  = (120, 120, 120)
        label  = "LOST"
        lcolor = (120, 120, 120)
    elif of_active:
        color  = (0, 220, 255)    # 노란색: OF 추적 중
        label  = "OF"
        lcolor = (0, 220, 255)
    else:
        color  = (255, 255, 255)  # 흰색: HSV 검출
        label  = "HSV"
        lcolor = (100, 255, 100)

    cv2.drawMarker(vis, (pcx, pcy), color,
                   cv2.MARKER_CROSS, markerSize=15, thickness=1,
                   line_type=cv2.LINE_AA)
    cv2.putText(vis, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, lcolor, 2, cv2.LINE_AA)

    return vis


def main():
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    tracker = LEDTrackerCA(
        dt=1/30,
        pos_noise=5e-4,
        vel_noise=0.8,
        acc_noise=0.1,
        meas_noise=0.3,
        max_missing=5,
        blend_alpha=BLEND_ALPHA,
        n_predict=N_PREDICT,
        decel_sens=DECEL_SENS,
    )
    prediction      = None
    in_predict_only = False
    redetect_count  = 0

    # ── 각도 로거 ─────────────────────────────────────────
    # (cx, cy)와 검출된 객체 사이의 단일 각도[deg]를 시간과 함께
    # angle_log.csv에 기록 (버퍼링 + 별도 스레드로 저자원 처리)
    # 실행 위치와 무관하게 항상 같은 곳에 저장되도록 절대경로 사용
    logger = AngleLogger("/home/pi/angle_log.csv")

    # ── Optical Flow 상태 ────────────────────────────────
    prev_red  = None    # 직전 프레임의 2R-G-B 채널
    of_point  = None    # 추적 중인 픽셀 좌표 (shape: [1,1,2] float32)
    of_active = False   # 현재 프레임이 OF로 추적된 상태인지

    print("  빨간 LED 서브픽셀 검출기 + CA 칼만 필터 (월드 각도 공간)")
    print(f"  blend_alpha={BLEND_ALPHA}  (0=correction, 1=prediction)")
    print(f"  decel_sens={DECEL_SENS}  (감속 감지 blend 억제 민감도)")
    print(f"  redetect_ramp={REDETECT_RAMP_FRAMES}프레임  (재검출 복귀 완충)")
    print(f"  optical flow: 2R-G-B 채널  noise_scale={OF_MEAS_NOISE_SCALE}  min_red={OF_MIN_RED}  fb_max={FB_MAX_ERR}px")
    print(f"  검출 실패 허용: {tracker.max_missing}프레임")
    print("  q / ESC : 종료")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패")
                break

            detections, mask = detect_red_led(frame)

            # ── 2R-G-B 채널 생성 (빨간색 강조, OF용) ──────────
            # 흰색(R≈G≈B→0), 파란색(R-B<0→0), 빨간 LED(2R-G-B≈최대)
            r = frame[:, :, 2].astype(np.int16)
            g = frame[:, :, 1].astype(np.int16)
            b = frame[:, :, 0].astype(np.int16)
            red_ch = np.clip(2 * r - g - b, 0, 255).astype(np.uint8)

            # ── 현재 서보 각도 읽기 ─────────────────────────────
            servo_yaw   = servo.yaw_angle
            servo_pitch = servo.pitch_angle

            # ── 검출 소스: HSV > OF > predict_only ──────────────
            px, py    = None, None
            of_active = False

            if detections:
                # ── HSV 검출 성공 ────────────────────────────────
                main_det = max(detections, key=lambda d: d['area'])
                px, py   = main_det['centroid']

                # OF 포인트를 HSV 결과로 갱신 (드리프트 방지)
                of_point = np.array([[px, py]], dtype=np.float32).reshape(1, 1, 2)

            elif prev_red is not None and of_point is not None and tracker.initialized:
                # ── HSV 실패 → Lucas-Kanade Optical Flow 시도 ───

                # Forward 추적
                new_pt, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_red, red_ch, of_point, None, **LK_PARAMS
                )

                of_valid = False
                if status is not None and status[0, 0] == 1:

                    # [검증 1] Forward-Backward 오차 체크
                    back_pt, status_b, _ = cv2.calcOpticalFlowPyrLK(
                        red_ch, prev_red, new_pt, None, **LK_PARAMS
                    )
                    fb_err = (np.linalg.norm(of_point[0, 0] - back_pt[0, 0])
                              if (status_b is not None and status_b[0, 0] == 1)
                              else 9999.0)

                    # [검증 2] 추적 위치의 2R-G-B 평균값 체크 (8px 반경 ROI)
                    npx, npy = int(round(new_pt[0, 0, 0])), int(round(new_pt[0, 0, 1]))
                    x1 = max(0, npx - 8); x2 = min(WIDTH,  npx + 8)
                    y1 = max(0, npy - 8); y2 = min(HEIGHT, npy + 8)
                    roi      = red_ch[y1:y2, x1:x2]
                    roi_mean = float(roi.mean()) if roi.size > 0 else 0.0

                    if fb_err <= FB_MAX_ERR and roi_mean >= OF_MIN_RED:
                        of_valid = True

                if of_valid:
                    px, py    = float(new_pt[0, 0, 0]), float(new_pt[0, 0, 1])
                    of_point  = new_pt
                    of_active = True
                else:
                    of_point = None   # 드리프트 또는 신뢰 불가 → 초기화

            # 다음 프레임을 위해 현재 채널 저장
            prev_red = red_ch

            # ── 칼만 업데이트 / predict_only 분기 ───────────────
            if px is not None:
                # (cx,cy)-객체 간 단일 각도를 로그에 기록 (non-blocking)
                logger.log(combined_angle(px, py))

                yaw_rel,   pitch_rel   = xy2angle.pixel_to_angles(px, py)
                yaw_world   = -servo_yaw   + yaw_rel
                pitch_world =  servo_pitch + pitch_rel

                if of_active:
                    # OF 측정: HSV보다 노이즈 높음 → measurementNoiseCov 일시 증가
                    orig_noise = tracker.kf.measurementNoiseCov.copy()
                    tracker.kf.measurementNoiseCov *= OF_MEAS_NOISE_SCALE

                prediction = tracker.update(yaw_world, pitch_world)

                if of_active:
                    tracker.kf.measurementNoiseCov = orig_noise

                if in_predict_only:
                    redetect_count = 0
                redetect_count  = min(redetect_count + 1, REDETECT_RAMP_FRAMES)
                in_predict_only = False

            else:
                # HSV, OF 모두 실패
                of_point = None
                if tracker.initialized:
                    prediction      = tracker.predict_only()
                    in_predict_only = True
                else:
                    prediction      = None
                    in_predict_only = False
                redetect_count = 0

            if prediction is not None:
                pred_yaw_w, pred_pitch_w, omega_yaw, omega_pitch, _, _ = prediction

                yaw_err   = pred_yaw_w   + servo_yaw
                pitch_err = pred_pitch_w - servo_pitch
                
                print(f"W=({pred_yaw_w:7.2f}, {pred_pitch_w:7.2f}) | "
                f"Target=({yaw_err:7.2f}, {pitch_err:7.2f}) | "
                f"Servo=({servo_yaw:7.2f}, {servo_pitch:7.2f})")

                if in_predict_only:
                    servo.move(yaw_err, pitch_err, use_d=False)

                else:
                    ramp = redetect_count / REDETECT_RAMP_FRAMES
                    servo.move(yaw_err * ramp, pitch_err * ramp,
                               vx_kalman=omega_yaw, vy_kalman=omega_pitch)


            vis      = draw_results(frame, prediction, of_active)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([vis, mask_bgr])

            stream_frame = cv2.resize(combined, (960, 360))
            with _frame_cond:
                global _latest_jpeg
                _latest_jpeg = _encode_jpeg(stream_frame)
                _frame_cond.notify_all()

    except KeyboardInterrupt:
        print("\n종료")
    finally:
        logger.close()   # 남은 버퍼 flush 후 로거 스레드 종료
        servo.stop()
        cap.release()


if __name__ == "__main__":
    flask_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True),
        daemon=True
    )
    flask_thread.start()
    print("  스트리밍 주소: http://<라즈베리파이IP>:5000")
    print("  종료하려면 Ctrl+C")
    main()
