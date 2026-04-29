import cv2
import numpy as np
import xy2angle
#from scc import ServoController

#servo = ServoController()

# ── 파라미터 ──────────────────────────────────────────────
CAM_ID       = 0
WIDTH        = 640
HEIGHT       = 480

H_LOW1,  S_LOW1,  V_LOW1  =   0, 90, 100
H_HIGH1, S_HIGH1, V_HIGH1 =  10, 255, 255
H_LOW2,  S_LOW2,  V_LOW2  = 160, 90, 100
H_HIGH2, S_HIGH2, V_HIGH2 = 180, 255, 255

MIN_AREA   = 1
MAX_AREA   = 500

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# ── 절충 비율 ─────────────────────────────────────────────
# 0.0 : 순수 correction (현재 프레임 최적 추정)
# 1.0 : 순수 prediction (다음 프레임 예측)
# 서보 레이턴시가 크면 값을 올리세요 (권장 범위: 0.2 ~ 0.5)
BLEND_ALPHA = 0.3


# ── 등가속 칼만 필터 (CA: Constant Acceleration) ─────────
class LEDTrackerCA:
    """
    상태 벡터: [x, y, vx, vy, ax, ay]
    측정 벡터: [x, y]
    등가속 운동 모델 기반 칼만 필터.

    update()는 correction과 prediction을 BLEND_ALPHA로 보간한
    위치를 반환합니다.
      α=0 → correction 기준 (안정적, 오버슈트 적음)
      α=1 → prediction 기준 (레이턴시 보상, 진동 위험)
    속도(vx, vy)는 항상 correction 기준으로 반환합니다.
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
        """
        측정값 (cx, cy)로 correction → blend → prediction 순으로 처리.

        반환: (blended_x, blended_y, vx, vy, ax, ay)
          - blended_x/y : correction과 prediction을 α로 보간한 위치
          - vx, vy      : correction 기준 속도 (PID D항에 사용)
        """
        measurement = np.array([[cx], [cy]], dtype=np.float32)

        if not self.initialized:
            self.kf.statePost = np.array(
                [[cx], [cy], [0.], [0.], [0.], [0.]], dtype=np.float32
            )
            self.initialized = True

        self.miss_count = 0

        # correction: 현재 프레임 최적 추정
        corrected = self.kf.correct(measurement)

        # prediction: 다음 프레임 위치 예측
        predicted = self.kf.predict()

        # 위치만 blend, 속도/가속도는 correction 기준 유지
        bx = (1 - self.blend_alpha) * corrected[0, 0] + self.blend_alpha * predicted[0, 0]
        by = (1 - self.blend_alpha) * corrected[1, 0] + self.blend_alpha * predicted[1, 0]

        return (
            bx, by,
            corrected[2, 0],   # vx — correction 기준
            corrected[3, 0],   # vy — correction 기준
            corrected[4, 0],   # ax
            corrected[5, 0],   # ay
        )

    def predict_only(self):
        """
        검출 실패 시 호출. 예측만 수행하고 miss_count 증가.
        max_missing 초과 시 트래커를 초기화하고 None 반환 (OUT).
        """
        self.miss_count += 1
        if self.miss_count > self.max_missing:
            print(f"[TRACKER] {self.max_missing}프레임 초과 누락 → 트래커 초기화 (OUT)")
            self.reset()
            return None

        predicted = self.kf.predict()
        return self._unpack(predicted)

    @staticmethod
    def _unpack(state):
        return (state[0, 0], state[1, 0],
                state[2, 0], state[3, 0],
                state[4, 0], state[5, 0])

    def reset(self):
        self.initialized = False
        self.miss_count  = 0


def subpixel_centroid(gray: np.ndarray, mask: np.ndarray):
    roi = gray.astype(np.float64)
    roi[mask == 0] = 0.0

    total = roi.sum()
    if total == 0:
        return None

    ys, xs = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    cx = (xs * roi).sum() / total
    cy = (ys * roi).sum() / total
    return cx, cy


def detect_red_led(frame: np.ndarray):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask1 = cv2.inRange(hsv,
                        (H_LOW1, S_LOW1, V_LOW1),
                        (H_HIGH1, S_HIGH1, V_HIGH1))
    mask2 = cv2.inRange(hsv,
                        (H_LOW2, S_LOW2, V_LOW2),
                        (H_HIGH2, S_HIGH2, V_HIGH2))
    mask  = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    results = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        component_mask = (labels == i).astype(np.uint8) * 255
        centroid = subpixel_centroid(gray, component_mask)
        if centroid is None:
            continue

        results.append({'centroid': centroid, 'area': area, 'bbox': (x, y, w, h)})

    return results, mask


def draw_results(frame: np.ndarray, detections: list, prediction: tuple):
    vis = frame.copy()

    if prediction is not None:
        px, py, vx, vy, ax, ay = prediction
        ipx, ipy = int(round(px)), int(round(py))

        cv2.drawMarker(vis, (ipx, ipy), (255, 255, 255),
                       cv2.MARKER_CROSS, markerSize=15, thickness=1,
                       line_type=cv2.LINE_AA)

        pred_label = (f"P=({px:.1f},{py:.1f}) "
                      f"V=({vx:.1f},{vy:.1f}) "
                      f"A=({ax:.1f},{ay:.1f})")
        cv2.putText(vis, pred_label,
                    (8, HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                    cv2.LINE_AA)
    return vis


def main():
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    tracker = LEDTrackerCA(
        dt=1.0,
        pos_noise=1e-2,
        vel_noise=1e-2,
        acc_noise=1e-1,
        meas_noise=1e-1,
        max_missing=5,
        blend_alpha=BLEND_ALPHA,   # correction ↔ prediction 절충 비율
    )
    prediction = None

    print("  빨간 LED 서브픽셀 검출기 + CA 칼만 필터 (correction/prediction blend)")
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

            # 픽셀 좌표 → 각도 오차
            yaw_err, pitch_err = xy2angle.pixel_to_angles(px, py)

            print(f"yaw_err={yaw_err:.4f}    pitch_err={pitch_err:.4f}")

            # PID 서보 제어 — 칼만 속도(vx, vy)를 D항에 활용
            #servo.move(yaw_err, pitch_err, vx_kalman=vx, vy_kalman=vy)

        vis      = draw_results(frame, detections, prediction)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([vis, mask_bgr])

        cv2.imshow("Red LED Detector  |  [original]  [mask]", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    servo.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()