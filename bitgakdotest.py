import cv2
import numpy as np
from sc import ServoController

servo = ServoController()

fx = 655.0
fy = 655.0
cx = 325.0
cy = 250.0

K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float64)

CENTER = (int(cx), int(cy))

# ── 칼만 필터 초기화 ──────────────────────────────────────────
kf = cv2.KalmanFilter(4, 2)

kf.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float32)

kf.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
], dtype=np.float32)

kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5
kf.errorCovPost        = np.eye(4, dtype=np.float32)

kf_initialized        = False
kf_lost_count         = 0
KF_LOST_MAX           = 5
last_kf_x, last_kf_y  = CENTER
# ─────────────────────────────────────────────────────────────

# ── 파라미터 ──────────────────────────────────────────────────
ABS_MIN_BRIGHTNESS = 200
SAT_MIN            = 1
SAT_MAX            = 100    # 형광등 공존 환경 대응
SHARP_RADIUS       = 10
SHARP_RATIO        = 5.0    # 형광등·태양광 구별 강화
BLOB_MIN_AREA      = 1
BLOB_MAX_AREA      = 80
FIXED_THRESHOLD    = 220    # 퍼센타일 대신 고정 임계값
# ─────────────────────────────────────────────────────────────

# ── [BUG 3 수정] SimpleBlobDetector 전역에서 한 번만 생성 ─────
_blob_params = cv2.SimpleBlobDetector_Params()
_blob_params.filterByArea        = True
_blob_params.minArea             = BLOB_MIN_AREA
_blob_params.maxArea             = BLOB_MAX_AREA
_blob_params.filterByColor       = True
_blob_params.blobColor           = 255
_blob_params.filterByCircularity = True
_blob_params.minCircularity      = 0.5
_blob_params.filterByConvexity   = False
_blob_params.filterByInertia     = True
_blob_params.minInertiaRatio     = 0.4
_blob_params.minThreshold        = 200
_blob_params.maxThreshold        = 255
_blob_params.thresholdStep       = 10
blob_detector = cv2.SimpleBlobDetector_create(_blob_params)
# ─────────────────────────────────────────────────────────────

def pixel_to_ray(u, v):
    ray = np.array([(u - cx) / fx,
                    (v - cy) / fy,
                    1.0])
    return ray / np.linalg.norm(ray)

def angle_between(ray1, ray2):
    r1_xz = np.array([ray1[0], 0, ray1[2]])
    r2_xz = np.array([ray2[0], 0, ray2[2]])
    r1_xz /= np.linalg.norm(r1_xz)
    r2_xz /= np.linalg.norm(r2_xz)
    cos_yaw = np.clip(np.dot(r1_xz, r2_xz), -1.0, 1.0)
    yaw = np.degrees(np.arccos(cos_yaw))
    yaw *= np.sign(ray2[0] - ray1[0])

    r1_yz = np.array([0, ray1[1], ray1[2]])
    r2_yz = np.array([0, ray2[1], ray2[2]])
    r1_yz /= np.linalg.norm(r1_yz)
    r2_yz /= np.linalg.norm(r2_yz)
    cos_pitch = np.clip(np.dot(r1_yz, r2_yz), -1.0, 1.0)
    pitch = np.degrees(np.arccos(cos_pitch))
    pitch *= np.sign(ray1[1] - ray2[1])

    return yaw, pitch

def subpixel_centroid(gray, cx_, cy_, r=2):
    x0 = max(0, cx_ - r)
    x1 = min(gray.shape[1], cx_ + r + 1)
    y0 = max(0, cy_ - r)
    y1 = min(gray.shape[0], cy_ + r + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    total = patch.sum()
    if total == 0:
        return float(cx_), float(cy_)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return float((xs * patch).sum() / total), float((ys * patch).sum() / total)

def is_sharp_peak(gray, cx_, cy_):
    cx_, cy_ = int(cx_), int(cy_)
    center_val = float(gray[cy_, cx_])

    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx_, cy_), SHARP_RADIUS, 255, 1)
    ring_pixels = gray[mask == 255]

    if len(ring_pixels) == 0:
        return False

    ring_mean = float(np.mean(ring_pixels))
    if ring_mean < 1:
        return True

    return (center_val / ring_mean) >= SHARP_RATIO

def detect_laser(gray):
    # ── Step 1: 포화 픽셀 수 확인 ─────────────────────────────
    saturated_mask = (gray >= 250).astype(np.uint8)
    sat_count = int(np.sum(saturated_mask))

    if sat_count < SAT_MIN:
        return None, "NO_SAT"

    # ── Step 2: 태양광 의심 구간 → connectedComponents ─────────
    if sat_count > SAT_MAX:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            saturated_mask, connectivity=8
        )

        best      = None
        best_area = 99999

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]

            if area > BLOB_MAX_AREA:
                continue

            cx_c = centroids[label][0]
            cy_c = centroids[label][1]

            if not is_sharp_peak(gray, cx_c, cy_c):
                continue

            if area < best_area:
                best_area = area
                best = (cx_c, cy_c)

        if best is None:
            return None, "SUN_NODET"

        sx, sy = subpixel_centroid(gray, int(best[0]), int(best[1]))
        return (sx, sy), "SUN_LASER"

    # ── Step 3: 일반 Blob 검출 ────────────────────────────────
    _, bright = cv2.threshold(gray, FIXED_THRESHOLD, 255, cv2.THRESH_BINARY)

    keypoints = blob_detector.detect(bright)  # [BUG 3 수정] 전역 detector 사용

    for kp in keypoints:
        bx, by = int(kp.pt[0]), int(kp.pt[1])
        if not is_sharp_peak(gray, bx, by):
            continue
        sx, sy = subpixel_centroid(gray, bx, by)
        return (sx, sy), "BLOB"

    # ── Fallback: 포화 픽셀 무게중심 ──────────────────────────
    ys, xs = np.where(saturated_mask.astype(bool))
    if len(xs) == 0:
        return None, "NONE"

    vals   = gray[ys, xs].astype(np.float32)
    total  = vals.sum()
    cx_sat = float((xs * vals).sum() / total)
    cy_sat = float((ys * vals).sum() / total)

    if not is_sharp_peak(gray, cx_sat, cy_sat):
        return None, "NONE"

    return (cx_sat, cy_sat), "SAT_CENTROID"


# ── 카메라 초기화 ─────────────────────────────────────────────
cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

ray_center = pixel_to_ray(*CENTER)
cv2.namedWindow("Angle Viewer")

prev_time = cv2.getTickCount()  # [BUG 1 수정] 0 대신 현재 틱으로 초기화

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cropped = frame[120:600, 320:960]
    display = cropped.copy()
    gray    = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    result, method = detect_laser(gray)

    cxx, cyy = None, None
    if result is not None:
        cxx, cyy = result

    # ── 칼만 필터 ─────────────────────────────────────────────
    if cxx is not None:
        kf_lost_count = 0

        if not kf_initialized:
            kf.statePre  = np.array([cxx, cyy, 0, 0], dtype=np.float32).reshape(4, 1)
            kf.statePost = np.array([cxx, cyy, 0, 0], dtype=np.float32).reshape(4, 1)
            kf_initialized = True

        kf.predict()
        measurement = np.array([[np.float32(cxx)], [np.float32(cyy)]])
        corrected   = kf.correct(measurement)
        kf_x, kf_y  = int(corrected[0]), int(corrected[1])
        last_kf_x, last_kf_y = kf_x, kf_y

    else:
        kf_lost_count += 1

        if kf_initialized and kf_lost_count <= KF_LOST_MAX:
            kf.statePost[2] *= 0.5
            kf.statePost[3] *= 0.5
            predicted = kf.predict()
            kf_x, kf_y = int(predicted[0]), int(predicted[1])
            last_kf_x, last_kf_y = kf_x, kf_y
        else:
            kf_initialized = False
            kf_lost_count  = 0

    # ── FPS ───────────────────────────────────────────────────
    curr_time = cv2.getTickCount()
    fps       = cv2.getTickFrequency() / (curr_time - prev_time)
    prev_time = curr_time

    # ── 시각화 ────────────────────────────────────────────────
    cv2.drawMarker(display, CENTER, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(display, "Center", (CENTER[0]+8, CENTER[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(display, f"FPS: {fps:.1f}", (540, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if kf_initialized:
        line_color   = (0, 255, 0)
        circle_color = (0, 255, 0)
    else:
        line_color   = (100, 100, 100)
        circle_color = (100, 100, 100)

    ray_p      = pixel_to_ray(last_kf_x, last_kf_y)
    yaw, pitch = angle_between(ray_center, ray_p)

    if cxx is not None:
        cv2.circle(display, (int(cxx), int(cyy)), 3, (0, 0, 255), 1)

    cv2.line(display, CENTER, (last_kf_x, last_kf_y), line_color, 1)
    cv2.circle(display, (last_kf_x, last_kf_y), 3, circle_color, 1)

    status = "TRACKING" if kf_initialized else "LOST"
    texts = [
        f"status   : {status}",
        f"method   : {method}",
        f"Raw      : ({int(cxx)}, {int(cyy)})" if cxx is not None else "Raw      : (---, ---)",
        f"Kalman   : ({last_kf_x}, {last_kf_y})",
        f"Yaw      : {yaw:+.2f} deg",
        f"Pitch    : {pitch:+.2f} deg",
    ]
    cv2.rectangle(display, (0, 395), (210, 495), (0, 0, 0), -1)
    for i, t in enumerate(texts):
        cv2.putText(display, t, (8, 410 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 180), 1)

    # ── 서보 제어 (필요 시 주석 해제) ────────────────────────
    if kf_initialized and cxx is not None:
        servo.move(yaw, pitch)

    cv2.imshow("Angle Viewer", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()