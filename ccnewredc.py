import cv2
import numpy as np
import xy2angle

# ── 파라미터 ──────────────────────────────────────────────
CAM_ID  = 0
WIDTH   = 640
HEIGHT  = 480

# ── RGB 빨간색 검출 파라미터 ──────────────────────────────
# 조건: R이 충분히 크고, R이 G/B보다 훨씬 클 것
R_MIN        = 170    # R 채널 최소값
R_MINUS_G    = 120     # R - G 최소 차이 (초록 억제)
R_MINUS_B    = 120     # R - B 최소 차이 (파랑 억제)

MIN_AREA = 1
MAX_AREA = 500

# 0.0 : 순수 correction (현재 프레임 최적 추정)
# 1.0 : 순수 prediction (다음 프레임 예측)
BLEND_ALPHA = 0.2


# ── 등가속 칼만 필터 (CA: Constant Acceleration) ─────────
class LEDTrackerCA:
    """
    상태 벡터: [x, y, vx, vy, ax, ay]
    측정 벡터: [x, y]
    등가속 운동 모델 기반 칼만 필터.
    """

    def __init__(self, dt: float = 1.0,
                 pos_noise: float = 1e-2,
                 vel_noise: float = 1e-2,
                 acc_noise: float = 1e-1,
                 meas_noise: float = 1e-1,
                 max_missing: int = 5,
                 blend_alpha: float = BLEND_ALPHA):
        self.kf = cv2.KalmanFilter(6, 2)
        self.initialized = False
        self.dt = dt
        self.max_missing = max_missing
        self.miss_count  = 0
        self.blend_alpha = blend_alpha

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

    def update(self, cx: float, cy: float):
        measurement = np.array([[cx], [cy]], dtype=np.float32)

        if not self.initialized:
            self.kf.statePost = np.array(
                [[cx], [cy], [0.], [0.], [0.], [0.]], dtype=np.float32
            )
            self.initialized = True

        self.miss_count = 0

        self.kf.predict()
        corrected = self.kf.correct(measurement)

        state_snap = self.kf.statePost.copy()
        cov_snap   = self.kf.errorCovPost.copy()
        next_predicted = self.kf.predict()
        self.kf.statePost    = state_snap
        self.kf.errorCovPost = cov_snap

        bx = (1 - self.blend_alpha) * corrected[0, 0] + self.blend_alpha * next_predicted[0, 0]
        by = (1 - self.blend_alpha) * corrected[1, 0] + self.blend_alpha * next_predicted[1, 0]

        return (
            bx, by,
            corrected[2, 0],
            corrected[3, 0],
            corrected[4, 0],
            corrected[5, 0],
        )

    def predict_only(self):
        self.miss_count += 1
        if self.miss_count > self.max_missing:
            self.reset()
            return None

        predicted = self.kf.predict()
        self.kf.statePost    = self.kf.statePre.copy()
        self.kf.errorCovPost = self.kf.errorCovPre.copy()

        vx, vy = predicted[2, 0], predicted[3, 0]
        print(f"[PREDICT_ONLY] miss={self.miss_count}  vx={vx:.3f}  vy={vy:.3f}")
        return self._unpack(predicted)

    @staticmethod
    def _unpack(state):
        return (state[0, 0], state[1, 0],
                state[2, 0], state[3, 0],
                state[4, 0], state[5, 0])

    def reset(self):
        self.initialized = False
        self.miss_count  = 0


# ── RGB 기반 빨간 LED 검출 ────────────────────────────────
def detect_red_led(frame: np.ndarray):
    """
    RGB 채널 비교로 빨간 LED를 검출합니다.

    조건:
      1. R >= R_MIN                  → 충분히 밝은 빨강
      2. R - G >= R_MINUS_G          → 초록 성분 억제
      3. R - B >= R_MINUS_B          → 파랑 성분 억제

    OpenCV는 BGR 순서이므로 채널 분리 시 주의.
    """
    # BGR → 채널 분리 (OpenCV 기본 순서)
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)

    # 세 조건을 모두 만족하는 픽셀만 마스크
    mask = (
        (r >= R_MIN) &
        ((r - g) >= R_MINUS_G) &
        ((r - b) >= R_MINUS_B)
    ).astype(np.uint8) * 255

    # 작은 점 살리기 (dilation)
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

        results.append({
            'centroid': (cx, cy),
            'area': area,
        })

    return results, mask


def draw_results(frame: np.ndarray, prediction: tuple):
    vis = frame.copy()
    if prediction is not None:
        px, py, vx, vy, ax, ay = prediction
        ipx, ipy = int(round(px)), int(round(py))
        cv2.drawMarker(vis, (ipx, ipy), (255, 255, 255),
                       cv2.MARKER_CROSS, markerSize=15, thickness=1,
                       line_type=cv2.LINE_AA)
    return vis


def open_camera():
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    return cap


def main():
    cap = open_camera()

    if cap is None or not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    tracker = LEDTrackerCA(
        dt=1,
        pos_noise=1e-1,
        vel_noise=5,
        acc_noise=1e-2,
        meas_noise=0.3,
        max_missing=5,
        blend_alpha=BLEND_ALPHA,
    )
    prediction = None

    print("  빨간 LED 검출기 (RGB 방식) + CA 칼만 필터")
    print(f"  R_MIN={R_MIN}  R-G>={R_MINUS_G}  R-B>={R_MINUS_B}")
    print(f"  blend_alpha={BLEND_ALPHA}  (0=correction, 1=prediction)")
    print(f"  검출 실패 허용: {tracker.max_missing}프레임")
    print("  q / ESC : 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break

        detections, mask = detect_red_led(frame)

        if detections:
            main_det = max(detections, key=lambda d: d['area'])
            cx, cy   = main_det['centroid']
            prediction = tracker.update(cx, cy)
        else:
            if tracker.initialized:
                prediction = tracker.predict_only()
            else:
                prediction = None

        if prediction is not None:
            px, py, vx, vy, ax, ay = prediction

            yaw_err, pitch_err = xy2angle.pixel_to_angles(px, py)

            vx_a = vx * (180 / np.pi) * 30 / xy2angle.getfx()
            vy_a = vy * (180 / np.pi) * 30 / xy2angle.getfx()
            #servo.move(yaw_err, pitch_err, vx_kalman=vx_a, vy_kalman=vy_a)

        vis      = draw_results(frame, prediction)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([vis, mask_bgr])

        cv2.imshow("Red LED Detector  |  [original]  [mask]", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    #servo.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()