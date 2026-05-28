import cv2
import numpy as np
import threading
import xy2angle
# sudo fuser -k /dev/video2
# from sccpid import ServoController
# servo = ServoController()

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

H_LOW1,  S_LOW1,  V_LOW1  =   0, 100, 100
H_HIGH1, S_HIGH1, V_HIGH1 =  10, 255, 255
H_LOW2,  S_LOW2,  V_LOW2  = 160, 100, 100
H_HIGH2, S_HIGH2, V_HIGH2 = 180, 255, 255

MIN_AREA   = 1
MAX_AREA   = 500

# 0.0 : 순수 correction (현재 프레임 최적 추정)
# 1.0 : 순수 prediction (다음 프레임 예측)
# 카메라가 움직이는 환경에서는 0.3~0.5 권장
BLEND_ALPHA = 0.3


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
    """

    def __init__(self, dt: float = 1/30,
                 pos_noise: float = 1e-2,
                 vel_noise: float = 5e-1,
                 acc_noise: float = 1e-2,
                 meas_noise: float = 0.3,
                 max_missing: int = 5,
                 blend_alpha: float = BLEND_ALPHA):
        self.kf = cv2.KalmanFilter(6, 2)
        self.initialized  = False
        self.dt           = dt
        self.max_missing  = max_missing
        self.miss_count   = 0
        self.blend_alpha  = blend_alpha
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

        # blend용: 다음 프레임 예측 후 상태 복원
        state_snap = self.kf.statePost.copy()
        cov_snap   = self.kf.errorCovPost.copy()
        next_pred  = self.kf.predict()
        self.kf.statePost    = state_snap
        self.kf.errorCovPost = cov_snap

        byaw   = (1 - self.blend_alpha) * corrected[0, 0] + self.blend_alpha * next_pred[0, 0]
        bpitch = (1 - self.blend_alpha) * corrected[1, 0] + self.blend_alpha * next_pred[1, 0]

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

        wy, wp = predicted[2, 0], predicted[3, 0]
        print(f"[PREDICT_ONLY] miss={self.miss_count}  "
              f"ω_yaw={wy:.3f}  ω_pitch={wp:.3f}")
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


def draw_results(frame: np.ndarray, prediction: tuple):
    """
    서보가 현재 바라보는 방향(화면 중심)에 조준 마커 표시.
    prediction이 있으면 흰색, 없으면 표시 안 함.
    """
    vis = frame.copy()

    if prediction is not None:
        cx_screen = WIDTH  // 2
        cy_screen = HEIGHT // 2
        cv2.drawMarker(vis, (cx_screen, cy_screen), (255, 255, 255),
                       cv2.MARKER_CROSS, markerSize=15, thickness=1,
                       line_type=cv2.LINE_AA)
    return vis


def main():
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    tracker = LEDTrackerCA(
        dt=1/30,
        pos_noise=1e-2,
        vel_noise=5e-1,
        acc_noise=1e-2,
        meas_noise=0.3,
        max_missing=5,
        blend_alpha=BLEND_ALPHA,
    )
    prediction = None

    # 서보 초기 각도 (servo 객체 없을 때 시뮬레이션용)
    servo_yaw   = 90.0
    servo_pitch = 90.0

    print("  빨간 LED 서브픽셀 검출기 + CA 칼만 필터 (월드 각도 공간)")
    print(f"  blend_alpha={BLEND_ALPHA}  (0=correction, 1=prediction)")
    print(f"  검출 실패 허용: {tracker.max_missing}프레임")
    print("  q / ESC : 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break

        detections, mask = detect_red_led(frame)

        # ── 현재 서보 각도 읽기 ─────────────────────────────
        # servo 객체 사용 시 아래 두 줄로 교체:
        # servo_yaw   = servo.yaw_angle
        # servo_pitch = servo.pitch_angle

        if detections:
            main_det = max(detections, key=lambda d: d['area'])
            px, py   = main_det['centroid']

            # 픽셀 → 카메라 기준 상대 각도
            yaw_rel, pitch_rel = xy2angle.pixel_to_angles(px, py)

            # 카메라 기준 상대각 → 월드 각도
            yaw_world   = servo_yaw   + yaw_rel
            pitch_world = servo_pitch + pitch_rel

            prediction = tracker.update(yaw_world, pitch_world)

        else:
            if tracker.initialized:
                prediction = tracker.predict_only()
            else:
                prediction = None

        if prediction is not None:
            pred_yaw_w, pred_pitch_w, omega_yaw, omega_pitch, _, _ = prediction

            # 서보 오차 = 예측된 월드 각도 - 현재 서보 각도
            yaw_err   = pred_yaw_w   - servo_yaw
            pitch_err = pred_pitch_w - servo_pitch

            #print(f"yaw_err={yaw_err:.3f}  pitch_err={pitch_err:.3f}  "
            #      f"ω_yaw={omega_yaw:.3f}  ω_pitch={omega_pitch:.3f}")

            # 각속도를 D항에 직접 사용 (deg/s 단위)
            # servo.move(yaw_err, pitch_err,
            #            vx_kalman=omega_yaw, vy_kalman=omega_pitch)

        vis      = draw_results(frame, prediction)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([vis, mask_bgr])

        # 1280×480 → 960×360 축소 후 인코딩 (전송 부하 감소)
        stream_frame = cv2.resize(combined, (960, 360))
        with _frame_cond:
            global _latest_jpeg
            _latest_jpeg = _encode_jpeg(stream_frame)
            _frame_cond.notify_all()

        # cv2.imshow("Red LED Detector  |  [original]  [mask]", combined)
        # key = cv2.waitKey(1) & 0xFF
        # if key in (ord('q'), 27):
        #     break

    # servo.stop()
    cap.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    flask_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True),
        daemon=True
    )
    flask_thread.start()
    print("  스트리밍 주소: http://<라즈베리파이IP>:5000")
    print("  종료하려면 Ctrl+C")
    main()