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

# ── 등가속 칼만 필터 (CA: Constant Acceleration) ─────────
class LEDTrackerCA:
    """
    상태 벡터: [x, y, vx, vy, ax, ay]
    측정 벡터: [x, y]
    등가속 운동 모델 기반 칼만 필터.

    processNoiseCov의 가속도 항(ax, ay)이 클수록
    가속도가 자주 바뀐다고 가정 → 측정값을 더 신뢰.
    """

    def __init__(self, dt: float = 1.0,
                 pos_noise: float = 1e-2,
                 vel_noise: float = 1e-2,
                 acc_noise: float = 1e-1,
                 meas_noise: float = 1e-1,
                 max_missing: int = 5):
        """
        dt          : 프레임 간격 (초). 실측 기반으로 넣어주면 더 정확.
        pos_noise   : 위치 프로세스 노이즈
        vel_noise   : 속도 프로세스 노이즈
        acc_noise   : 가속도 프로세스 노이즈 (클수록 빠르게 반응)
        meas_noise  : 측정 노이즈 (클수록 측정값을 덜 신뢰)
        max_missing : 검출 실패를 허용할 최대 프레임 수.
                      이 값을 초과하면 트래커를 초기화(OUT).
        """
        self.kf = cv2.KalmanFilter(6, 2)  # 상태 6차원, 측정 2차원
        self.initialized = False
        self.dt = dt
        self.max_missing  = max_missing   # 허용 최대 누락 프레임
        self.miss_count   = 0             # 현재 연속 누락 프레임 수

        # ── 전이 행렬 (등가속 운동) ───────────────────────
        # x_new  = x  + vx*dt + 0.5*ax*dt²
        # vx_new = vx + ax*dt
        # ax_new = ax
        dt2 = 0.5 * dt ** 2
        self.kf.transitionMatrix = np.array([
            [1, 0, dt,  0, dt2,   0],
            [0, 1,  0, dt,   0, dt2],
            [0, 0,  1,  0,  dt,   0],
            [0, 0,  0,  1,   0,  dt],
            [0, 0,  0,  0,   1,   0],
            [0, 0,  0,  0,   0,   1],
        ], dtype=np.float32)

        # ── 측정 행렬 (x, y만 관측) ──────────────────────
        self.kf.measurementMatrix = np.zeros((2, 6), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0
        self.kf.measurementMatrix[1, 1] = 1.0

        # ── 프로세스 노이즈 공분산 ───────────────────────
        self.kf.processNoiseCov = np.diag([
            pos_noise, pos_noise,
            vel_noise, vel_noise,
            acc_noise, acc_noise,
        ]).astype(np.float32)

        # ── 측정 노이즈 공분산 ───────────────────────────
        self.kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * meas_noise
        )

        # ── 초기 오차 공분산 ─────────────────────────────
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

    def update(self, cx: float, cy: float):
        """
        측정값 (cx, cy) 로 필터 보정 후 다음 프레임 위치 예측.
        검출 성공 시 miss_count 리셋.
        반환: (pred_x, pred_y, vx, vy, ax, ay)
        """
        measurement = np.array([[cx], [cy]], dtype=np.float32)

        if not self.initialized:
            self.kf.statePost = np.array(
                [[cx], [cy], [0.], [0.], [0.], [0.]], dtype=np.float32
            )
            self.initialized = True

        self.miss_count = 0               # 검출 성공 → 누락 카운트 리셋
        self.kf.correct(measurement)
        predicted = self.kf.predict()
        return self._unpack(predicted)

    def predict_only(self):
        """
        검출 실패 시 호출. 예측만 수행하고 miss_count 증가.
        max_missing 초과 시 트래커를 초기화하고 None 반환 (OUT).
        반환: (pred_x, pred_y, vx, vy, ax, ay) 또는 None
        """
        self.miss_count += 1
        if self.miss_count > self.max_missing:
            print(f"[TRACKER] {self.max_missing}프레임 초과 누락 → 트래커 초기화 (OUT)")
            self.reset()
            return None

        predicted = self.kf.predict()
        return self._unpack(predicted)

    @staticmethod
    def _unpack(predicted):
        return (predicted[0, 0], predicted[1, 0],
                predicted[2, 0], predicted[3, 0],
                predicted[4, 0], predicted[5, 0])

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
    """
    {'centroid': (cx, cy), 'area': int, 'bbox': (x,y,w,h)}
    """
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

def draw_results(frame: np.ndarray, detections: list,
                 prediction: tuple):
    vis = frame.copy()

    if prediction is not None:
        px, py, vx, vy, ax, ay = prediction
        ipx, ipy = int(round(px)), int(round(py))

        cv2.drawMarker(vis, (ipx, ipy), (0, 0, 255),
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
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # ── 칼만 필터 초기화 ─────────────────────────────────
    # acc_noise를 높이면 급격한 가속 변화에 빠르게 반응.
    # meas_noise를 높이면 측정값보다 예측 모델을 더 신뢰.
    tracker = LEDTrackerCA(dt=1.0,
                           pos_noise=1e-2,
                           vel_noise=1e-2,
                           acc_noise=1e-1,
                           meas_noise=1e-1,
                           max_missing=5)   # 5프레임 초과 누락 시 OUT
    prediction = None   # 현재 프레임의 예측값

    print("  빨간 LED 서브픽셀 검출기 + CA 칼만 필터")
    print(f"  검출 실패 허용: {tracker.max_missing}프레임")
    print("  q / ESC : 종료")
    print("  RED     : 칼만 예측 위치")

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

            """
            px, py, vx, vy, ax, ay = prediction
            print(f"[LED] meas=({cx:.4f},{cy:.4f})  "
                  f"pred=({px:.4f},{py:.4f})  "
                  f"v=({vx:.2f},{vy:.2f})  "
                  f"a=({ax:.2f},{ay:.2f})")
            """

        else:
            if tracker.initialized:
                remaining = tracker.max_missing - tracker.miss_count
                prediction = tracker.predict_only()
                """
                if prediction is None:
                    print("[LED] 검출 실패 누적 한도 초과 → OUT")
                else:
                    print(f"[LED] 검출 실패 — 칼만 예측 유지 "
                          f"({tracker.miss_count}/{tracker.max_missing}프레임)")
                """
            else:
                prediction = None

        if prediction is not None:
            yaw, pitch = xy2angle.pixel_to_angles(prediction[0], prediction[1])
            print(f"yaw={yaw:.4f}    pitch={pitch:.4f}")
            #servo.move(yaw, pitch)

        vis      = draw_results(frame, detections, prediction)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([vis, mask_bgr])

        cv2.imshow("Red LED Detector  |  [original]  [mask]", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()