"""
nnewredcr_logonly.py
────────────────────────────────────────────────────────────
모터를 전혀 구동하지 않는 "로깅 전용" 버전.

원본(nnewredcr.py)에서 서보 제어에 필요했던 부분
  - ServoController / PID
  - 등가속 칼만 필터(LEDTrackerCA)
  - Optical Flow 재검출 로직
을 전부 제거하고, 아래 흐름만 남겼다.

  1. 카메라에서 프레임을 읽는다.
  2. HSV 기반으로 빨간 LED를 검출한다.
  3. 검출되면 angle_logger.AngleLogger.log_pixel()로
     (time_s, angle_deg, yaw_deg, pitch_deg)를 CSV에 기록한다.
  4. 검출 결과를 표시한 프레임을 웹으로 스트리밍한다.

모터가 움직이지 않으므로 여기서 기록되는 각도는 "현재 서보 각도 대비
오차"가 아니라, angle_logger.py의 정의 그대로 "화면 중심(광축) 기준
타겟의 상대 각도"이다.
"""

import cv2
import numpy as np
import threading
import time
import xy2angle
from angle_logger import AngleLogger
# sudo fuser -k /dev/video2
from flask import Flask, Response, jsonify

app = Flask(__name__)

# ── 영상 스트리밍 최적화 ─────────────────────────────────────
# 메인 루프에서 JPEG 인코딩을 직접 수행하면 imencode()가 끝날 때까지
# 루프가 멈춘다. 따라서 메인 루프는 "가장 최신 표시용 프레임"만 넘기고,
# resize + JPEG 인코딩은 별도 스레드가 담당한다.
#
# 프레임 큐를 쌓지 않고 최신 1장만 유지하므로 네트워크/브라우저가 느려져도
# 과거 프레임이 누적되어 화면이 수백 ms~수 초씩 뒤처지는 현상을 줄인다.
STREAM_WIDTH        = 480
STREAM_HEIGHT       = 360
STREAM_FPS          = 25.0   # 표시 전용 FPS. 검출 FPS와 독립
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


def draw_results(frame: np.ndarray, detected: bool, centroid: tuple = None):
    """
    화면 중심(주점)에 조준 마커 + ±5도 정사각형을 항상 표시.
    - 검출됨: 흰색 마커 + 'DETECTED' (녹색), 실제 검출 좌표에도 점 표시
    - 검출 안 됨: 회색 마커 + 'LOST' (회색)
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
    if detected:
        color  = (255, 255, 255)
        label  = "DETECTED"
        lcolor = (100, 255, 100)
        if centroid is not None:
            cv2.drawMarker(vis, (int(round(centroid[0])), int(round(centroid[1]))),
                           (0, 220, 255), cv2.MARKER_CROSS, markerSize=12,
                           thickness=1, line_type=cv2.LINE_AA)
    else:
        color  = (120, 120, 120)
        label  = "LOST"
        lcolor = (120, 120, 120)

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
            cmd = input("검출/로깅을 시작하려면 'start' 입력 후 Enter: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if cmd == 'start':
            start_event.set()
            return

        print("  'start'를 입력해야 검출/로깅이 시작됩니다.")


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

    print('로깅 전용 모드 (모터 구동 없음).')
    print('start 전에도 웹 카메라 미리보기는 동작합니다.')
    print('웹의 카메라 OFF 버튼을 누르면 draw/resize/JPEG/전송 작업을 중단합니다.')

    logger = None
    tracking_active = False

    next_stream_t = 0.0
    stream_period = 1.0 / max(STREAM_FPS, 1.0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('프레임 읽기 실패')
                break

            # start 전: 카메라 미리보기만. 검출/로거는 동작하지 않는다.
            if not tracking_active:
                if _stream_enabled.is_set():
                    now_stream = time.monotonic()
                    if now_stream >= next_stream_t:
                        _submit_stream_frame(frame)
                        next_stream_t = now_stream + stream_period

                if not tracking_start.is_set():
                    continue

                # start가 들어온 뒤에만 파일을 생성한다.
                # 모터를 움직이지 않으므로 검출된 원시 각도만 그대로 기록한다.
                logger = AngleLogger('/home/pi/angle_log.csv')
                tracking_active = True

                print('검출 및 angle logger를 시작합니다.')

            detections = detect_red_led(frame)

            detected = False
            centroid = None
            if detections:
                main_det = max(detections, key=lambda d: d['area'])
                centroid = main_det['centroid']
                px, py = centroid
                logger.log_pixel(px, py)
                detected = True

            # OFF면 draw/copy/resize/JPEG/네트워크용 프레임 제출을 전부 생략한다.
            if _stream_enabled.is_set():
                now_stream = time.monotonic()
                if now_stream >= next_stream_t:
                    vis = draw_results(frame, detected, centroid)
                    _submit_stream_frame(vis)
                    next_stream_t = now_stream + stream_period

    except KeyboardInterrupt:
        print('\n종료')
    finally:
        input_stop.set()
        if logger is not None:
            logger.close()
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