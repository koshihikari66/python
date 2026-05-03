import cv2
import numpy as np
import xy2angle
#sudo fuser -k /dev/video2

from sccpid import ServoController

servo = ServoController()

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
# 서보 레이턴시가 크면 값을 올리세요 (권장 범위: 0.2 ~ 0.5)
BLEND_ALPHA = 0.22


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
        측정값 (cx, cy)로 predict → correct (표준 순서) 처리.

        반환: (blended_x, blended_y, vx, vy, ax, ay)
          - blended_x/y : correction과 next_prediction을 α로 보간한 위치
          - vx, vy      : correction 기준 속도 (PID D항에 사용)

        blend 구현:
          correct 후 상태를 스냅샷 → predict로 다음 프레임 예측 → 상태 복원.
          내부 상태는 항상 correction 사후 상태(statePost)로 유지된다.
        """
        measurement = np.array([[cx], [cy]], dtype=np.float32)

        if not self.initialized:
            self.kf.statePost = np.array(
                [[cx], [cy], [0.], [0.], [0.], [0.]], dtype=np.float32
            )
            self.initialized = True

        self.miss_count = 0

        # 1. predict: 사전 추정 (표준 순서)
        self.kf.predict()

        # 2. correct: 측정값으로 사후 추정
        corrected = self.kf.correct(measurement)

        # 3. 다음 프레임 예측 (blend용) — 상태 스냅샷 후 복원
        state_snap = self.kf.statePost.copy()
        cov_snap   = self.kf.errorCovPost.copy()

        next_predicted = self.kf.predict()

        self.kf.statePost    = state_snap   # 내부 상태를 correction으로 되돌림
        self.kf.errorCovPost = cov_snap

        # 위치만 blend, 속도/가속도는 correction 기준 유지
        bx = (1 - self.blend_alpha) * corrected[0, 0] + self.blend_alpha * next_predicted[0, 0]
        by = (1 - self.blend_alpha) * corrected[1, 0] + self.blend_alpha * next_predicted[1, 0]

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
            #print(f"[TRACKER] {self.max_missing}프레임 초과 누락 → 트래커 초기화 (OUT)")
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



def detect_red_led(frame: np.ndarray):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv,
                        (H_LOW1, S_LOW1, V_LOW1),
                        (H_HIGH1, S_HIGH1, V_HIGH1))
    mask2 = cv2.inRange(hsv,
                        (H_LOW2, S_LOW2, V_LOW2),
                        (H_HIGH2, S_HIGH2, V_HIGH2))
    mask = cv2.bitwise_or(mask1, mask2)

    # 작은 점 살리기 (dilation)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # contour 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [], mask

    results = []

    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        # 중심 계산
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
    vis = frame.copy()

    if prediction is not None:
        px, py, vx, vy, ax, ay = prediction
        ipx, ipy = int(round(px)), int(round(py))

        cv2.drawMarker(vis, (ipx, ipy), (255, 255, 255),
                       cv2.MARKER_CROSS, markerSize=15, thickness=1,
                       line_type=cv2.LINE_AA)
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

            yaw_err, pitch_err = xy2angle.pixel_to_angles(px, py)

            #print(f"yaw_err={yaw_err:.4f}    pitch_err={pitch_err:.4f}")

            # PID 서보 제어 — 칼만 속도(vx, vy)를 D항에 활용
            # vx, vy -> angle velocity
            vx=vx/xy2angle.getfx()
            vy=vy/xy2angle.getfx()
            servo.move(yaw_err, pitch_err, vx_kalman=vx, vy_kalman=vy)
            #servo.move(yaw_err, pitch_err)

        vis      = draw_results(frame, prediction)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([vis, mask_bgr])

        cv2.imshow("Red LED Detector  |  [original]  [mask]", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

################################################################
    servo.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()