import cv2
import numpy as np
import xy2angle
# sudo fuser -k /dev/video2
# from sccpid import ServoController
# servo = ServoController()

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

# 0.0 : 순수 correction
# 1.0 : 순수 prediction
# 카메라가 움직이는 환경: 0.3~0.5 권장
BLEND_ALPHA = 0.3


# ── 등가속 칼만 필터 (월드 각도 공간) ────────────────────
class LEDTrackerCA:
    """
    상태 벡터: [yaw_world, pitch_world, ω_yaw, ω_pitch, α_yaw, α_pitch]
    측정 벡터: [yaw_world, pitch_world]

    월드 각도 = servo.actual_yaw + yaw_rel (1차 지연 모델 적용)

    수정 이력:
    - 월드 각도 공간으로 전환
    - dt=1/30 실제 초 단위 사용
    - _was_missing 플래그로 복귀 첫 프레임 이중예측 버그 수정
    - servo.actual_yaw 사용으로 명령값-실제값 오염 제거
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
        self._was_missing = False

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

        복귀 첫 프레임(_was_missing=True)은 predict() 생략.
        statePost가 이미 최신 예측값이므로 이중예측 방지.

        반환: (pred_yaw, pred_pitch, ω_yaw, ω_pitch, α_yaw, α_pitch)
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
            corrected[2, 0],   # ω_yaw   [deg/s]
            corrected[3, 0],   # ω_pitch [deg/s]
            corrected[4, 0],   # α_yaw
            corrected[5, 0],   # α_pitch
        )

    def predict_only(self):
        """
        검출 실패 시 호출. 예측만 수행하고 miss_count 증가.
        max_missing 초과 시 트래커 초기화 후 None 반환.
        statePost = statePre 복사로 연속 miss 체이닝 유지.
        _was_missing = True 로 복귀 첫 프레임 이중예측 방지.
        """
        self.miss_count += 1
        if self.miss_count > self.max_missing:
            self.reset()
            return None

        predicted = self.kf.predict()
        self.kf.statePost    = self.kf.statePre.copy()
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

    # ── servo 객체 없을 때 시뮬레이션용 ──────────────────
    # servo 사용 시 아래 블록 전체를 삭제하고
    # servo = ServoController() 주석 해제
    class _DummyServo:
        actual_yaw   = 90.0
        actual_pitch = 90.0
    servo = _DummyServo()
    # ─────────────────────────────────────────────────────

    print("  빨간 LED 서브픽셀 검출기 + CA 칼만 필터")
    print("  (월드 각도 공간 | 1차 지연 모델 적용)")
    print(f"  blend_alpha={BLEND_ALPHA}  검출 실패 허용: {tracker.max_missing}프레임")
    print("  q / ESC : 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break

        detections, mask = detect_red_led(frame)

        # ── 1차 지연 모델 기반 추정 실제 서보 위치 ────────
        # 명령값(yaw_angle)이 아닌 추정 실제값(actual_yaw)을 사용
        # → 명령-실제 간 차이로 인한 월드각 오염 방지
        servo_yaw   = servo.actual_yaw
        servo_pitch = servo.actual_pitch

        if detections:
            main_det = max(detections, key=lambda d: d['area'])
            px, py   = main_det['centroid']

            # 픽셀 → 카메라 기준 상대 각도
            yaw_rel, pitch_rel = xy2angle.pixel_to_angles(px, py)

            # 추정 실제 서보 위치 기준 월드 각도
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

            # 서보 오차 = 예측된 월드 각도 - 추정 실제 서보 각도
            yaw_err   = pred_yaw_w   - servo_yaw
            pitch_err = pred_pitch_w - servo_pitch

            print(f"yaw_err={yaw_err:.3f}  pitch_err={pitch_err:.3f}  "
                  f"ω_yaw={omega_yaw:.3f}  ω_pitch={omega_pitch:.3f}")

            # servo.move(yaw_err, pitch_err,
            #            vx_kalman=omega_yaw, vy_kalman=omega_pitch)

        vis      = draw_results(frame, prediction)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([vis, mask_bgr])

        cv2.imshow("Red LED Detector  |  [original]  [mask]", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    # servo.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()