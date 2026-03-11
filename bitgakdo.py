import cv2
import numpy as np
#from sc import ServoController

#servo = ServoController()

fx = 655.0
fy = 655.0
cx = 325.0
cy = 250.0
prev_time = 0

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

cap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 1
params.maxArea = 30
params.filterByColor = True
params.blobColor = 255
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = True
params.minInertiaRatio = 0.5
params.minThreshold = 200
params.maxThreshold = 230
params.thresholdStep = 10
detector = cv2.SimpleBlobDetector_create(params)

ABS_MIN_BRIGHTNESS    = 160
BRIGHT_PERCENTILE     = 99.5
VALID_BRIGHTNESS_SMALL = 160
VALID_BRIGHTNESS_BIG  = 255
SMALL_BLOB_SIZE       = 3

ray_center = pixel_to_ray(*CENTER)
cv2.namedWindow("Angle Viewer")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cropped = frame[120:600, 320:960]
    display = cropped.copy()
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    cxx, cyy = None, None
    method = "NONE"

    # ── BLOB 검출 ────────────────────────────────────────────
    _, maxVal, _, _ = cv2.minMaxLoc(gray)
    if maxVal >= ABS_MIN_BRIGHTNESS:
        thresh_val = max(np.percentile(gray, BRIGHT_PERCENTILE), ABS_MIN_BRIGHTNESS)
        _, bright = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        keypoints = detector.detect(bright)

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            brightness = gray[y, x]
            if kp.size <= SMALL_BLOB_SIZE:
                if brightness >= VALID_BRIGHTNESS_SMALL:
                    cxx, cyy = x, y
                    method = "BLOB"
                    break
            else:
                if brightness >= VALID_BRIGHTNESS_BIG:
                    cxx, cyy = x, y
                    method = "BLOB"
                    break

    if cxx is None:
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)
        if maxVal >= VALID_BRIGHTNESS_SMALL:
            cxx, cyy = maxLoc
            method = "MAXPIX"

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
            # last_kf_x, last_kf_y 유지 → 마지막 위치 고정

    # ── FPS ──────────────────────────────────────────────────
    curr_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (curr_time - prev_time)
    prev_time = curr_time

    # ── 시각화 ───────────────────────────────────────────────
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
        color_raw = (0, 0, 255) if method == "BLOB" else (255, 0, 0)
        cv2.circle(display, (cxx, cyy), 1, color_raw, 1)

    cv2.line(display, CENTER, (last_kf_x, last_kf_y), line_color, 1)
    cv2.circle(display, (last_kf_x, last_kf_y), 1, circle_color, 1)

    status = "TRACKING" if kf_initialized else "LOST"
    texts = [
        f"status   : {status}",
        f"method   : {method}",
        f"Raw      : ({cxx}, {cyy})" if cxx is not None else "Raw      : (---, ---)",
        f"Kalman   : ({last_kf_x}, {last_kf_y})",
        f"Yaw      : {yaw:+.2f} deg",
        f"Pitch    : {pitch:+.2f} deg",
    ]
    cv2.rectangle(display, (0, 395), (210, 495), (0, 0, 0), -1)
    for i, t in enumerate(texts):
        cv2.putText(display, t, (8, 410 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 180), 1)

    # if kf_initialized and cxx is not None:
    #     servo.move(yaw, pitch)

    cv2.imshow("Angle Viewer", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()