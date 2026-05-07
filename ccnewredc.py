import cv2
import numpy as np
import xy2angle

# from sccpid import ServoController
# servo = ServoController()

# ── 파라미터 ──────────────────────────────────────────────
CAM_ID = 0

WIDTH  = 640
HEIGHT = 480

MIN_AREA = 1
MAX_AREA = 500

# 빨강 dominance threshold
R_MIN      = 140
RG_DIFF    = 50
RB_DIFF    = 50

# 0.0 : correction 중심
# 1.0 : prediction 중심
BLEND_ALPHA = 0.2


# ── 등가속 칼만 필터 ─────────────────────────────────────
class LEDTrackerCA:

    def __init__(
        self,
        dt: float = 1.0,
        pos_noise: float = 1e-2,
        vel_noise: float = 1e-2,
        acc_noise: float = 1e-1,
        meas_noise: float = 1e-1,
        max_missing: int = 5,
        blend_alpha: float = BLEND_ALPHA,
    ):

        self.kf = cv2.KalmanFilter(6, 2)

        self.initialized = False
        self.max_missing = max_missing
        self.miss_count  = 0
        self.blend_alpha = blend_alpha

        dt2 = 0.5 * dt ** 2

        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.zeros((2, 6), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0
        self.kf.measurementMatrix[1, 1] = 1.0

        self.kf.processNoiseCov = np.diag([
            pos_noise, pos_noise,
            vel_noise, vel_noise,
            acc_noise, acc_noise
        ]).astype(np.float32)

        self.kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * meas_noise
        )

        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

    def update(self, cx: float, cy: float):

        measurement = np.array([[cx], [cy]], dtype=np.float32)

        if not self.initialized:
            self.kf.statePost = np.array([
                [cx],
                [cy],
                [0.],
                [0.],
                [0.],
                [0.]
            ], dtype=np.float32)

            self.initialized = True

        self.miss_count = 0

        # predict
        self.kf.predict()

        # correct
        corrected = self.kf.correct(measurement)

        # next prediction for blend
        state_snap = self.kf.statePost.copy()
        cov_snap   = self.kf.errorCovPost.copy()

        next_predicted = self.kf.predict()

        self.kf.statePost    = state_snap
        self.kf.errorCovPost = cov_snap

        bx = (
            (1 - self.blend_alpha) * corrected[0, 0]
            + self.blend_alpha * next_predicted[0, 0]
        )

        by = (
            (1 - self.blend_alpha) * corrected[1, 0]
            + self.blend_alpha * next_predicted[1, 0]
        )

        return (
            bx,
            by,
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

        return self._unpack(predicted)

    @staticmethod
    def _unpack(state):

        return (
            state[0, 0],
            state[1, 0],
            state[2, 0],
            state[3, 0],
            state[4, 0],
            state[5, 0],
        )

    def reset(self):

        self.initialized = False
        self.miss_count  = 0


# ── 빨간 LED 검출 ─────────────────────────────────────────
def detect_red_led(frame: np.ndarray):

    # BGR 채널 분리
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)

    # 빨강 dominance 기반 threshold
    mask = (
        (r > R_MIN) &
        (r > g + RG_DIFF) &
        (r > b + RB_DIFF)
    ).astype(np.uint8) * 255

    # Connected Components
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(mask, connectivity=8)

    results = []

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        cx, cy = centroids[i]

        results.append({
            'centroid': (cx, cy),
            'area': area,
            'bbox': (x, y, w, h)
        })

    return results, mask


# ── 시각화 ────────────────────────────────────────────────
def draw_results(frame: np.ndarray, prediction: tuple):

    vis = frame.copy()

    if prediction is not None:

        px, py, vx, vy, ax, ay = prediction

        ipx = int(round(px))
        ipy = int(round(py))

        cv2.drawMarker(
            vis,
            (ipx, ipy),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            markerSize=15,
            thickness=1,
            line_type=cv2.LINE_AA
        )

    return vis


# ── 메인 ──────────────────────────────────────────────────
def main():

    cap = cv2.VideoCapture(CAM_ID)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

    cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    if not cap.isOpened():
        print("카메라 열기 실패")
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

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        detections, mask = detect_red_led(frame)

        # ── detection 성공 ─────────────────────────
        if detections:

            main_det = max(
                detections,
                key=lambda d: d['area']
            )

            cx, cy = main_det['centroid']

            prediction = tracker.update(cx, cy)

        # ── detection 실패 ─────────────────────────
        else:

            if tracker.initialized:
                prediction = tracker.predict_only()
            else:
                prediction = None

        # ── 서보 제어 ──────────────────────────────
        if prediction is not None:

            px, py, vx, vy, ax, ay = prediction

            yaw_err, pitch_err = \
                xy2angle.pixel_to_angles(px, py)

            vx = vx * 30 / xy2angle.getfx()
            vy = vy * 30 / xy2angle.getfx()

            # servo.move(
            #     yaw_err,
            #     pitch_err,
            #     vx_kalman=vx,
            #     vy_kalman=vy
            # )

        # ── 디스플레이 ─────────────────────────────
        vis = draw_results(frame, prediction)

        cv2.imshow("tracking", vis)
        cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):
            break

    # servo.stop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()