import cv2
import numpy as np
import threading
import time
import xy2angle
from angle_logger import AngleLogger
# sudo fuser -k /dev/video2
from sccpid_first_orderr import ServoController

# ── YAW / PITCH 공통 서보 파라미터 ────────────────────────
# 두 축이 같은 PID 값을 쓰므로 설정값은 하나로 합친다.
# I 게인과 EMA 스무딩은 기존 설정값이 0이라 실제 영향이 없어 제거했다.
PID_KP = 0.13
PID_KD = 0.001
PID_OUTPUT_LIMIT = 3.0
PID_DEADBAND = 1.0
SERVO_MAX_SPEED = 180.0

HOME_STEP_DEG = 6.0
HOME_STEP_DELAY = 0.1

servo = ServoController(
    kp=PID_KP,
    kd=PID_KD,
    output_limit=PID_OUTPUT_LIMIT,
    deadband=PID_DEADBAND,
    max_speed=SERVO_MAX_SPEED,
    home_step_deg=HOME_STEP_DEG,
    home_step_delay=HOME_STEP_DELAY,
)

from flask import Flask, Response, jsonify

app = Flask(__name__)

# ── 영상 스트리밍 최적화 ─────────────────────────────────────
# 추적/서보 제어 루프에서 JPEG 인코딩을 직접 수행하면 imencode()가 끝날 때까지
# 메인 루프가 멈춘다. 따라서 메인 루프는 "가장 최신 표시용 프레임"만 넘기고,
# resize + JPEG 인코딩은 별도 스레드가 담당한다.
#
# 프레임 큐를 쌓지 않고 최신 1장만 유지하므로 네트워크/브라우저가 느려져도
# 과거 프레임이 누적되어 화면이 수백 ms~수 초씩 뒤처지는 현상을 줄인다.
STREAM_WIDTH        = 480
STREAM_HEIGHT       = 360
STREAM_FPS          = 25.0   # 표시 전용 FPS. 추적/서보 제어 FPS와 독립
STREAM_JPEG_QUALITY = 55     # 낮을수록 CPU/네트워크 부하 감소

_raw_frame_cond   = threading.Condition()
_latest_raw_frame = None
_raw_frame_seq    = -1

_jpeg_cond   = threading.Condition()
_latest_jpeg = None
_jpeg_seq    = -1

_stream_stop = threading.Event()
_stream_enabled = threading.Event()
_stream_enabled.set()  # 기본 ON: start 전에도 카메라 미리보기 표시


def _encode_jpeg(frame: np.ndarray, quality: int = STREAM_JPEG_QUALITY) -> bytes | None:
    ok, buf = cv2.imencode(
        '.jpg', frame,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        return None
    return buf.tobytes()


def _submit_stream_frame(frame: np.ndarray):
    """웹 스트리밍이 ON일 때만 최신 프레임을 인코더에 넘긴다."""
    if not _stream_enabled.is_set():
        return

    global _latest_raw_frame, _raw_frame_seq
    with _raw_frame_cond:
        _latest_raw_frame = frame
        _raw_frame_seq += 1
        _raw_frame_cond.notify()


def _stream_encoder_loop():
    """resize/JPEG 인코딩 전용 스레드. 항상 가장 최신 프레임만 처리."""
    global _latest_jpeg, _jpeg_seq

    last_raw_seq = -1
    min_interval = 1.0 / max(STREAM_FPS, 1.0)
    next_encode_t = 0.0

    while not _stream_stop.is_set():
        with _raw_frame_cond:
            _raw_frame_cond.wait_for(
                lambda: (
                    _stream_stop.is_set()
                    or (_stream_enabled.is_set() and _raw_frame_seq != last_raw_seq)
                ),
                timeout=0.5,
            )
            if _stream_stop.is_set():
                break
            if not _stream_enabled.is_set():
                continue

            frame = _latest_raw_frame
            seq = _raw_frame_seq

        if frame is None:
            continue

        # 표시 FPS 제한. 기다리는 동안 새 프레임은 이전 것을 덮어쓰므로 backlog 없음.
        now = time.monotonic()
        if now < next_encode_t:
            if _stream_stop.wait(next_encode_t - now):
                break

            # 기다리는 동안 더 최신 프레임이 들어왔으면 그 프레임으로 교체.
            with _raw_frame_cond:
                if _raw_frame_seq != seq:
                    frame = _latest_raw_frame
                    seq   = _raw_frame_seq

        if not _stream_enabled.is_set():
            last_raw_seq = seq
            continue

        resized = cv2.resize(
            frame, (STREAM_WIDTH, STREAM_HEIGHT),
            interpolation=cv2.INTER_AREA
        )
        jpeg = _encode_jpeg(resized)
        if jpeg is None:
            last_raw_seq = seq
            next_encode_t = time.monotonic() + min_interval
            continue

        with _jpeg_cond:
            _latest_jpeg = jpeg
            _jpeg_seq += 1
            _jpeg_cond.notify_all()

        last_raw_seq = seq
        next_encode_t = time.monotonic() + min_interval


def _mjpeg_generator():
    """Flask 스트리밍 제너레이터 — 새 JPEG가 생길 때만 최신 1장을 전송."""
    last_jpeg_seq = _jpeg_seq

    while _stream_enabled.is_set():
        with _jpeg_cond:
            _jpeg_cond.wait_for(
                lambda: (
                    _stream_stop.is_set()
                    or not _stream_enabled.is_set()
                    or _jpeg_seq != last_jpeg_seq
                ),
                timeout=1.0,
            )
            if _stream_stop.is_set() or not _stream_enabled.is_set():
                return

            jpeg = _latest_jpeg
            seq = _jpeg_seq

        if jpeg is None:
            continue

        last_jpeg_seq = seq
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n'
            b'Cache-Control: no-cache\r\n\r\n' + jpeg + b'\r\n'
        )


@app.route('/video')
def video_feed():
    if not _stream_enabled.is_set():
        return ('stream off', 503)
    return Response(
        _mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
        }
    )


@app.post('/stream/on')
def stream_on():
    _stream_enabled.set()
    with _raw_frame_cond:
        _raw_frame_cond.notify_all()
    return jsonify(enabled=True)


@app.post('/stream/off')
def stream_off():
    global _latest_raw_frame, _latest_jpeg
    _stream_enabled.clear()
    with _raw_frame_cond:
        _latest_raw_frame = None
        _raw_frame_cond.notify_all()
    with _jpeg_cond:
        _latest_jpeg = None
        _jpeg_cond.notify_all()
    return jsonify(enabled=False)


@app.route('/')
def index():
    return r'''
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera Preview</title>
  <style>
    body { font-family: sans-serif; margin: 18px; background: #111; color: #eee; }
    button { font-size: 16px; padding: 9px 16px; margin-right: 8px; }
    #state { margin-left: 8px; font-weight: 700; }
    img { display: block; max-width: 100%; margin-top: 16px; border: 1px solid #444; }
  </style>
</head>
<body>
  <button onclick="setStream(true)">카메라 ON</button>
  <button onclick="setStream(false)">카메라 OFF</button>
  <span id="state">ON</span>
  <img id="cam" src="/video">
<script>
async function setStream(on) {
  await fetch(on ? '/stream/on' : '/stream/off', {method: 'POST'});
  const img = document.getElementById('cam');
  document.getElementById('state').textContent = on ? 'ON' : 'OFF';
  if (on) {
    img.style.display = 'block';
    img.src = '/video?t=' + Date.now();
  } else {
    img.src = '';
    img.style.display = 'none';
  }
}
</script>
</body>
</html>
'''

# ── 파라미터 ──────────────────────────────────────────────
CAM_ID       = 0
WIDTH        = 640
HEIGHT       = 480
CAM_FPS      = 30

H_LOW1,  S_LOW1,  V_LOW1  =   0, 140, 140
H_HIGH1, S_HIGH1, V_HIGH1 =  12, 255, 255
H_LOW2,  S_LOW2,  V_LOW2  = 168, 140, 140
H_HIGH2, S_HIGH2, V_HIGH2 = 180, 255, 255

MIN_AREA   = 1
MAX_AREA   = 500

# detect_red_led() 안에서 매 프레임 생성하지 않도록 1회만 할당.
_DILATE_KERNEL = np.ones((3, 3), np.uint8)

# 2R-G-B 채널 계산용 OpenCV transform 행렬.
# 기존 int16 배열 3개(r/g/b)를 매 프레임 만드는 NumPy 연산과 같은 결과를
# OpenCV 내부 연산으로 계산하여 메모리 할당량을 줄인다.
_RED_TRANSFORM = np.array([[-1.0, -1.0, 2.0]], dtype=np.float32)

# 0.0 : 순수 correction (현재 프레임 최적 추정)
# 1.0 : 순수 prediction (다음 프레임 예측)
# 카메라가 움직이는 환경에서는 0.3~0.5 권장
BLEND_ALPHA = 0.9   # 0.0=correction만, 1.0=prediction만 (반드시 0~1)

# 몇 프레임 앞을 예측할지 — 이 값으로 레이턴시 보상량 조절
# 서보 지연 ≈ N프레임이면 N으로 설정 (권장: 2~5)
N_PREDICT   = 2

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
    - predict_only()에도 blend 적용하여 update()↔predict_only() 전환 시 타겟 위치 점프 제거
    """

    def __init__(self, dt: float = 1/30,
                 pos_noise: float = 1e-2,
                 vel_noise: float = 5,
                 acc_noise: float = 0.5,
                 meas_noise: float = 0.05,
                 max_missing: int = 5,
                 blend_alpha: float = BLEND_ALPHA,
                 n_predict: int = N_PREDICT):
        self.kf = cv2.KalmanFilter(6, 2)
        self.initialized  = False
        self.max_missing  = max_missing
        self.miss_count   = 0
        self.blend_alpha  = blend_alpha
        self.n_predict    = n_predict
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

        반환: (pred_yaw, pred_pitch, ω_yaw, ω_pitch)
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

        # ── blend용: n_predict 프레임 앞 예측 후 상태 복원 ───────
        if self.blend_alpha > 0:
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

            byaw   = (1 - self.blend_alpha) * corrected[0, 0] + self.blend_alpha * next_pred[0, 0]
            bpitch = (1 - self.blend_alpha) * corrected[1, 0] + self.blend_alpha * next_pred[1, 0]
        else:
            byaw   = corrected[0, 0]
            bpitch = corrected[1, 0]

        return (
            byaw, bpitch,
            corrected[2, 0],   # ω_yaw   — correction 기준 각속도 [deg/s]
            corrected[3, 0],   # ω_pitch — correction 기준 각속도 [deg/s]
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

            byaw   = (1 - self.blend_alpha) * predicted[0, 0] + self.blend_alpha * next_pred[0, 0]
            bpitch = (1 - self.blend_alpha) * predicted[1, 0] + self.blend_alpha * next_pred[1, 0]
            return (byaw, bpitch,
                    predicted[2, 0], predicted[3, 0])

        return self._unpack(predicted)

    @staticmethod
    def _unpack(state):
        return (state[0, 0], state[1, 0],
                state[2, 0], state[3, 0])

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

    mask = cv2.dilate(mask, _DILATE_KERNEL, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

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
        results.append({
            'centroid': (cx, cy),
            'area': area,
        })

    return results


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


def _start_command_loop(start_event: threading.Event, stop_event: threading.Event):
    """input() 때문에 카메라 루프가 멈추지 않도록 start 명령만 별도 스레드에서 받는다."""
    while not start_event.is_set() and not stop_event.is_set():
        try:
            cmd = input("트래킹을 시작하려면 'start' 입력 후 Enter: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if cmd == 'start':
            start_event.set()
            return

        print("  'start'를 입력해야 트래킹이 시작됩니다.")


def main():
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    if not cap.isOpened():
        print('카메라를 열 수 없습니다.')
        return

    tracking_start = threading.Event()
    input_stop = threading.Event()
    threading.Thread(
        target=_start_command_loop,
        args=(tracking_start, input_stop),
        name='start-command',
        daemon=True,
    ).start()

    print('서보 초기화 완료 (90/90).')
    print('start 전에도 웹 카메라 미리보기는 동작합니다.')
    print('웹의 카메라 OFF 버튼을 누르면 draw/resize/JPEG/전송 작업을 중단합니다.')

    tracker = None
    logger = None
    tracking_active = False

    prediction = None
    in_predict_only = False
    redetect_count = 0
    prev_red = None
    of_point = None
    of_active = False

    next_stream_t = 0.0
    stream_period = 1.0 / max(STREAM_FPS, 1.0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('프레임 읽기 실패')
                break

            # start 전: 카메라 미리보기만. 추적/PID/로거는 동작하지 않는다.
            if not tracking_active:
                if _stream_enabled.is_set():
                    now_stream = time.monotonic()
                    if now_stream >= next_stream_t:
                        _submit_stream_frame(frame)
                        next_stream_t = now_stream + stream_period

                if not tracking_start.is_set():
                    continue

                tracker = LEDTrackerCA(
                    dt=1 / 30,
                    pos_noise=5e-4,
                    vel_noise=0.8,
                    acc_noise=0.1,
                    meas_noise=0.3,
                    max_missing=5,
                    blend_alpha=BLEND_ALPHA,
                    n_predict=N_PREDICT,
                )

                # start가 들어온 뒤에만 파일/스레드를 생성한다.
                logger = AngleLogger('/home/pi/angle_log.csv')
                tracking_active = True
                prediction = None
                in_predict_only = False
                redetect_count = 0
                prev_red = None
                of_point = None
                of_active = False

                print('트래킹 및 angle logger를 시작합니다.')
                print(f'  blend_alpha={BLEND_ALPHA}')
                print(f'  redetect_ramp={REDETECT_RAMP_FRAMES}프레임')
                print(
                    '  optical flow: 2R-G-B 채널  '
                    f'noise_scale={OF_MEAS_NOISE_SCALE}  '
                    f'min_red={OF_MIN_RED}  fb_max={FB_MAX_ERR}px'
                )
                print(f'  검출 실패 허용: {tracker.max_missing}프레임')

            detections = detect_red_led(frame)
            red_ch = cv2.transform(frame, _RED_TRANSFORM)

            servo_yaw = servo.yaw_angle
            servo_pitch = servo.pitch_angle

            px, py = None, None
            of_active = False

            if detections:
                main_det = max(detections, key=lambda d: d['area'])
                px, py = main_det['centroid']
                of_point = np.array([[px, py]], dtype=np.float32).reshape(1, 1, 2)

            elif prev_red is not None and of_point is not None and tracker.initialized:
                new_pt, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_red, red_ch, of_point, None, **LK_PARAMS
                )

                of_valid = False
                if status is not None and status[0, 0] == 1:
                    back_pt, status_b, _ = cv2.calcOpticalFlowPyrLK(
                        red_ch, prev_red, new_pt, None, **LK_PARAMS
                    )
                    fb_err = (
                        np.linalg.norm(of_point[0, 0] - back_pt[0, 0])
                        if status_b is not None and status_b[0, 0] == 1
                        else 9999.0
                    )

                    npx = int(round(new_pt[0, 0, 0]))
                    npy = int(round(new_pt[0, 0, 1]))
                    x1, x2 = max(0, npx - 8), min(WIDTH, npx + 8)
                    y1, y2 = max(0, npy - 8), min(HEIGHT, npy + 8)
                    roi = red_ch[y1:y2, x1:x2]
                    roi_mean = float(roi.mean()) if roi.size > 0 else 0.0
                    of_valid = fb_err <= FB_MAX_ERR and roi_mean >= OF_MIN_RED

                if of_valid:
                    px = float(new_pt[0, 0, 0])
                    py = float(new_pt[0, 0, 1])
                    of_point = new_pt
                    of_active = True
                else:
                    of_point = None

            prev_red = red_ch

            if px is not None:
                logger.log_pixel(px, py)

                yaw_rel, pitch_rel = xy2angle.pixel_to_angles(px, py)
                yaw_world = -servo_yaw + yaw_rel
                pitch_world = servo_pitch + pitch_rel

                if of_active:
                    orig_noise = tracker.kf.measurementNoiseCov.copy()
                    tracker.kf.measurementNoiseCov *= OF_MEAS_NOISE_SCALE

                prediction = tracker.update(yaw_world, pitch_world)

                if of_active:
                    tracker.kf.measurementNoiseCov = orig_noise

                if in_predict_only:
                    redetect_count = 0
                redetect_count = min(redetect_count + 1, REDETECT_RAMP_FRAMES)
                in_predict_only = False

            else:
                of_point = None
                if tracker.initialized:
                    prediction = tracker.predict_only()
                    in_predict_only = True
                else:
                    prediction = None
                    in_predict_only = False
                redetect_count = 0

            if prediction is not None:
                pred_yaw_w, pred_pitch_w, omega_yaw, omega_pitch = prediction
                yaw_err = pred_yaw_w + servo_yaw
                pitch_err = pred_pitch_w - servo_pitch

                if in_predict_only:
                    servo.move(yaw_err, pitch_err, use_d=False)
                else:
                    ramp = redetect_count / REDETECT_RAMP_FRAMES
                    servo.move(
                        yaw_err * ramp,
                        pitch_err * ramp,
                        vx_kalman=omega_yaw,
                        vy_kalman=omega_pitch,
                    )

            # OFF면 draw/copy/resize/JPEG/네트워크용 프레임 제출을 전부 생략한다.
            if _stream_enabled.is_set():
                now_stream = time.monotonic()
                if now_stream >= next_stream_t:
                    vis = draw_results(frame, prediction, of_active)
                    _submit_stream_frame(vis)
                    next_stream_t = now_stream + stream_period

    except KeyboardInterrupt:
        print('\n종료')
    finally:
        input_stop.set()
        if logger is not None:
            logger.close()
        servo.stop()
        cap.release()


if __name__ == "__main__":
    encoder_thread = threading.Thread(
        target=_stream_encoder_loop,
        name="stream-encoder",
        daemon=True
    )
    encoder_thread.start()

    flask_thread = threading.Thread(
        target=lambda: app.run(
            host='0.0.0.0',
            port=5000,
            threaded=True,
            use_reloader=False
        ),
        daemon=True
    )
    flask_thread.start()

    print("  스트리밍 주소: http://<라즈베리파이IP>:5000")
    print("  웹에서 카메라 ON/OFF 가능")
    print("  종료하려면 Ctrl+C")

    try:
        main()
    finally:
        _stream_stop.set()
        with _raw_frame_cond:
            _raw_frame_cond.notify_all()
        with _jpeg_cond:
            _jpeg_cond.notify_all()
