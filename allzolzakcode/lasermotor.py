import cv2
import numpy as np
import threading
import time
import xy2angle
from angle_logger import AngleLogger

# sudo fuser -k /dev/video2
from sccpid_laser import ServoController

# ── YAW / PITCH 공통 서보 파라미터 ────────────────────────
PID_KP = 0.5
PID_KD = 0.00
PID_OUTPUT_LIMIT = 4.0
PID_DEADBAND = 0.5
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
STREAM_WIDTH = 480
STREAM_HEIGHT = 360
STREAM_FPS = 25.0
STREAM_JPEG_QUALITY = 55
_raw_frame_cond = threading.Condition()
_latest_raw_frame = None
_raw_frame_seq = -1
_jpeg_cond = threading.Condition()
_latest_jpeg = None
_jpeg_seq = -1
_stream_stop = threading.Event()
_stream_enabled = threading.Event()
_stream_enabled.set()


def _encode_jpeg(frame: np.ndarray, quality: int = STREAM_JPEG_QUALITY) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return None
    return buf.tobytes()


def _submit_stream_frame(frame: np.ndarray):
    if not _stream_enabled.is_set():
        return
    global _latest_raw_frame, _raw_frame_seq
    with _raw_frame_cond:
        _latest_raw_frame = frame
        _raw_frame_seq += 1
        _raw_frame_cond.notify()


def _stream_encoder_loop():
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
        now = time.monotonic()
        if now < next_encode_t:
            if _stream_stop.wait(next_encode_t - now):
                break
            with _raw_frame_cond:
                if _raw_frame_seq != seq:
                    frame = _latest_raw_frame
                    seq = _raw_frame_seq
        if not _stream_enabled.is_set():
            last_raw_seq = seq
            continue
        resized = cv2.resize(
            frame, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_AREA
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
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n\r\n" + jpeg + b"\r\n"
        )


@app.route("/video")
def video_feed():
    if not _stream_enabled.is_set():
        return ("stream off", 503)
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.post("/stream/on")
def stream_on():
    _stream_enabled.set()
    with _raw_frame_cond:
        _raw_frame_cond.notify_all()
    return jsonify(enabled=True)


@app.post("/stream/off")
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


@app.route("/")
def index():
    return r"""
<!doctype html>
<html lang="ko">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>Camera Preview</title>

<style>

body {
    font-family: sans-serif;
    margin: 18px;
    background: #111;
    color: #eee;
}

button {
    font-size: 16px;
    padding: 9px 16px;
    margin-right: 8px;
}

#state {
    margin-left: 8px;
    font-weight: 700;
}

img {
    display: block;
    max-width: 100%;
    margin-top: 16px;
    border: 1px solid #444;
}

</style>

</head>

<body>

<button onclick="setStream(true)">
    카메라 ON
</button>

<button onclick="setStream(false)">
    카메라 OFF
</button>

<span id="state">
    ON
</span>

<img id="cam" src="/video">

<script>

async function setStream(on) {

    await fetch(
        on ? '/stream/on' : '/stream/off',
        {method: 'POST'}
    );

    const img =
        document.getElementById('cam');

    document.getElementById('state')
        .textContent = on ? 'ON' : 'OFF';

    if (on) {

        img.style.display = 'block';

        img.src =
            '/video?t=' + Date.now();

    } else {

        img.src = '';

        img.style.display = 'none';

    }

}

</script>

</body>
</html>
"""


# ── 카메라 파라미터 ───────────────────────────────────────
CAM_ID = 0
WIDTH = 640
HEIGHT = 480
CAM_FPS = 30
# ── RED LED HSV 범위 ──────────────────────────────────────
H_LOW1, S_LOW1, V_LOW1 = 0, 80, 140
H_HIGH1, S_HIGH1, V_HIGH1 = 12, 255, 255
H_LOW2, S_LOW2, V_LOW2 = 168, 80, 140
H_HIGH2, S_HIGH2, V_HIGH2 = 180, 255, 255
MIN_AREA = 1
MAX_AREA = 500
_DILATE_KERNEL = np.ones((3, 3), np.uint8)
# ── RED 강조 채널 ─────────────────────────────────────────
_RED_TRANSFORM = np.array([[-1.0, -1.0, 2.0]], dtype=np.float32)
# ── Kalman 설정 ───────────────────────────────────────────
BLEND_ALPHA = 0
N_PREDICT = 0
REDETECT_RAMP_FRAMES = 2
# ── Optical Flow 파라미터 ─────────────────────────────────
LK_PARAMS = dict(
    winSize=(11, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03,
    ),
)
OF_MEAS_NOISE_SCALE = 3.0
OF_MIN_RED = 14
FB_MAX_ERR = 4.0


# ── 등가속 칼만 필터 ──────────────────────────────────────
class LEDTrackerCA:
    """
    상태 벡터:
    [yaw_world,
     pitch_world,
     omega_yaw,
     omega_pitch,
     alpha_yaw,
     alpha_pitch]
    측정 벡터:
    [yaw_world,
     pitch_world]
    """

    def __init__(
        self,
        dt: float = 1 / 30,
        pos_noise: float = 1e-2,
        vel_noise: float = 5,
        acc_noise: float = 0.5,
        meas_noise: float = 0.05,
        max_missing: int = 5,
        blend_alpha: float = BLEND_ALPHA,
        n_predict: int = N_PREDICT,
    ):
        self.kf = cv2.KalmanFilter(6, 2)
        self.initialized = False
        self.max_missing = max_missing
        self.miss_count = 0
        self.blend_alpha = blend_alpha
        self.n_predict = n_predict
        self._was_missing = False
        dt2 = 0.5 * dt**2
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, dt, 0, dt2, 0],
                [0, 1, 0, dt, 0, dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.zeros((2, 6), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0
        self.kf.measurementMatrix[1, 1] = 1.0
        self.kf.processNoiseCov = np.diag(
            [
                pos_noise,
                pos_noise,
                vel_noise,
                vel_noise,
                acc_noise,
                acc_noise,
            ]
        ).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

    def update(self, yaw_world: float, pitch_world: float):
        measurement = np.array([[yaw_world], [pitch_world]], dtype=np.float32)
        if not self.initialized:
            self.kf.statePost = np.array(
                [
                    [yaw_world],
                    [pitch_world],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                ],
                dtype=np.float32,
            )
            self.initialized = True
            self._was_missing = False
        self.miss_count = 0
        if not self._was_missing:
            self.kf.predict()
        self._was_missing = False
        corrected = self.kf.correct(measurement)
        if self.blend_alpha > 0:
            state_snap = self.kf.statePost.copy()
            cov_snap = self.kf.errorCovPost.copy()
            state_pre_snap = self.kf.statePre.copy()
            cov_pre_snap = self.kf.errorCovPre.copy()
            next_pred = corrected
            for _ in range(self.n_predict):
                next_pred = self.kf.predict()
            self.kf.statePost = state_snap
            self.kf.errorCovPost = cov_snap
            self.kf.statePre = state_pre_snap
            self.kf.errorCovPre = cov_pre_snap
            byaw = (1 - self.blend_alpha) * corrected[
                0, 0
            ] + self.blend_alpha * next_pred[0, 0]
            bpitch = (1 - self.blend_alpha) * corrected[
                1, 0
            ] + self.blend_alpha * next_pred[1, 0]
        else:
            byaw = corrected[0, 0]
            bpitch = corrected[1, 0]
        return (byaw, bpitch, corrected[2, 0], corrected[3, 0])

    def predict_only(self):
        self.miss_count += 1
        if self.miss_count > self.max_missing:
            self.reset()
            return None
        predicted = self.kf.predict()
        self.kf.statePost = self.kf.statePre.copy()
        self.kf.errorCovPost = self.kf.errorCovPre.copy()
        self._was_missing = True
        if self.blend_alpha > 0 and self.n_predict > 1:
            state_snap = self.kf.statePost.copy()
            cov_snap = self.kf.errorCovPost.copy()
            next_pred = predicted
            for _ in range(self.n_predict - 1):
                next_pred = self.kf.predict()
            self.kf.statePost = state_snap
            self.kf.errorCovPost = cov_snap
            byaw = (1 - self.blend_alpha) * predicted[
                0, 0
            ] + self.blend_alpha * next_pred[0, 0]
            bpitch = (1 - self.blend_alpha) * predicted[
                1, 0
            ] + self.blend_alpha * next_pred[1, 0]
            return (byaw, bpitch, predicted[2, 0], predicted[3, 0])
        return self._unpack(predicted)

    @staticmethod
    def _unpack(state):
        return (state[0, 0], state[1, 0], state[2, 0], state[3, 0])

    def reset(self):
        self.initialized = False
        self.miss_count = 0
        self._was_missing = False


# ── RED LED 검출 ───────────────────────────────────────────
def detect_red_led(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (H_LOW1, S_LOW1, V_LOW1), (H_HIGH1, S_HIGH1, V_HIGH1))
    mask2 = cv2.inRange(hsv, (H_LOW2, S_LOW2, V_LOW2), (H_HIGH2, S_HIGH2, V_HIGH2))
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
        results.append(
            {
                "centroid": (cx, cy),
                "area": area,
            }
        )
    return results


# ── 화면 표시 ─────────────────────────────────────────────
def draw_results(frame: np.ndarray, prediction: tuple, of_active: bool = False):
    vis = frame.copy()
    pcx = int(xy2angle.getcx())
    pcy = int(xy2angle.getcy())
    # ±5도 정사각형
    deg4 = np.radians(5.0)
    dx = int(round(xy2angle.getfx() * np.tan(deg4)))
    dy = dx
    cv2.rectangle(
        vis, (pcx - dx, pcy - dy), (pcx + dx, pcy + dy), (0, 220, 220), 1, cv2.LINE_AA
    )
    if prediction is None:
        color = (120, 120, 120)
        label = "LOST"
        lcolor = (120, 120, 120)
    elif of_active:
        color = (0, 220, 255)
        label = "OF"
        lcolor = (0, 220, 255)
    else:
        color = (255, 255, 255)
        label = "HSV"
        lcolor = (100, 255, 100)
    cv2.drawMarker(
        vis,
        (pcx, pcy),
        color,
        cv2.MARKER_CROSS,
        markerSize=15,
        thickness=1,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, lcolor, 2, cv2.LINE_AA
    )
    return vis


# ── START 명령 ────────────────────────────────────────────
def _start_command_loop(start_event: threading.Event, stop_event: threading.Event):
    while not start_event.is_set() and not stop_event.is_set():
        try:
            cmd = (
                input("트래킹을 시작하려면 " "'start' 입력 후 Enter: ").strip().lower()
            )
        except (EOFError, KeyboardInterrupt):
            return
        if cmd == "start":
            start_event.set()
            return
        print("  'start'를 입력해야 " "트래킹이 시작됩니다.")


# ── MAIN ──────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 255)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    tracking_start = threading.Event()
    input_stop = threading.Event()
    threading.Thread(
        target=_start_command_loop,
        args=(tracking_start, input_stop),
        name="start-command",
        daemon=True,
    ).start()
    print("서보 초기화 완료 (90/90).")
    print("start 전에도 웹 카메라 " "미리보기는 동작합니다.")
    print("웹의 카메라 OFF 버튼을 누르면 " "draw/resize/JPEG/전송 작업을 중단합니다.")
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
                print("프레임 읽기 실패")
                break
            # start 전
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
                logger = AngleLogger("/home/pi/angle_log.csv")
                tracking_active = True
                prediction = None
                in_predict_only = False
                redetect_count = 0
                prev_red = None
                of_point = None
                of_active = False
                print("트래킹 및 angle logger를 " "시작합니다.")
                print(f"  blend_alpha={BLEND_ALPHA}")
                print(f"  redetect_ramp=" f"{REDETECT_RAMP_FRAMES}프레임")
                print(
                    "  optical flow: "
                    "2R-G-B 채널  "
                    f"noise_scale="
                    f"{OF_MEAS_NOISE_SCALE}  "
                    f"min_red="
                    f"{OF_MIN_RED}  "
                    f"fb_max="
                    f"{FB_MAX_ERR}px"
                )
                print(f"  검출 실패 허용: " f"{tracker.max_missing}프레임")
            # ── LED 검출 ────────────────────────────────
            detections = detect_red_led(frame)
            red_ch = cv2.transform(frame, _RED_TRANSFORM)
            servo_yaw = servo.yaw_angle
            servo_pitch = servo.pitch_angle
            px, py = None, None
            of_active = False
            # ── HSV 검출 성공 ───────────────────────────
            if detections:
                main_det = max(detections, key=lambda d: d["area"])
                px, py = main_det["centroid"]
                of_point = np.array([[px, py]], dtype=np.float32).reshape(1, 1, 2)
            # ── Optical Flow ───────────────────────────
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
                        if (status_b is not None and status_b[0, 0] == 1)
                        else 9999.0
                    )
                    npx = int(round(new_pt[0, 0, 0]))
                    npy = int(round(new_pt[0, 0, 1]))
                    x1 = max(0, npx - 8)
                    x2 = min(WIDTH, npx + 8)
                    y1 = max(0, npy - 8)
                    y2 = min(HEIGHT, npy + 8)
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
            # ── 픽셀 → 각도 → 월드 좌표 ────────────────
            if px is not None:
                logger.log_pixel(px, py)
                yaw_rel, pitch_rel = xy2angle.pixel_to_angles(px, py)
                # ==================================================
                # ★ 카메라 180° 물리 회전 보정
                #
                # 카메라가 180° 뒤집혀 장착되었기 때문에
                # 이미지상의 좌우/상하 방향이 모두 반대가 됨.
                #
                # 따라서 상대 yaw/pitch의 부호를 반전.
                # ==================================================
                yaw_rel = -yaw_rel
                pitch_rel = -pitch_rel
                # 기존 월드 좌표 변환
                yaw_world = -servo_yaw + yaw_rel
                pitch_world = servo_pitch + pitch_rel
                # ── Optical Flow 측정 노이즈 증가 ──────
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
            # ── LED 검출 실패 ──────────────────────────
            else:
                of_point = None
                if tracker.initialized:
                    prediction = tracker.predict_only()
                    in_predict_only = True
                else:
                    prediction = None
                    in_predict_only = False
                redetect_count = 0
            # ── 서보 제어 ──────────────────────────────
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
            # ── 영상 스트리밍 ──────────────────────────
            if _stream_enabled.is_set():
                now_stream = time.monotonic()
                if now_stream >= next_stream_t:
                    vis = draw_results(frame, prediction, of_active)
                    _submit_stream_frame(vis)
                    next_stream_t = now_stream + stream_period
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        input_stop.set()
        if logger is not None:
            logger.close()
        servo.stop()
        cap.release()


# ── 실행 ──────────────────────────────────────────────────
if __name__ == "__main__":
    encoder_thread = threading.Thread(
        target=_stream_encoder_loop, name="stream-encoder", daemon=True
    )
    encoder_thread.start()
    flask_thread = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0", port=5000, threaded=True, use_reloader=False
        ),
        daemon=True,
    )
    flask_thread.start()
    print("  스트리밍 주소: " "http://<라즈베리파이IP>:5000")
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