import cv2
import numpy as np

cap = cv2.VideoCapture(cv2.CAP_DSHOW+0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -9)
cap.set(cv2.CAP_PROP_GAIN, 0)

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 1
params.maxArea = 100

params.filterByColor = True
params.blobColor = 255

params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

params.minThreshold = 200
params.maxThreshold = 255
params.thresholdStep = 10

detector = cv2.SimpleBlobDetector_create(params)

# =========================
# Tunables
# =========================
ABS_MIN_BRIGHTNESS = 160
BRIGHT_PERCENTILE = 99.5

VALID_BRIGHTNESS_SMALL = 160
VALID_BRIGHTNESS_BIG = 230
SMALL_BLOB_SIZE = 3

# =========================
# Main loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cxx, cyy = None, None
    method = "NONE"

    # -------------------------
    # 1️⃣ Blob path
    # -------------------------
    _, maxVal, _, _ = cv2.minMaxLoc(gray)

    if maxVal >= ABS_MIN_BRIGHTNESS:
        thresh_val = max(
            np.percentile(gray, BRIGHT_PERCENTILE),
            ABS_MIN_BRIGHTNESS
        )

        _, bright = cv2.threshold(
            gray, thresh_val, 255, cv2.THRESH_BINARY
        )

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

    # -------------------------
    # 2️⃣ Fallback: Max pixel
    # -------------------------
    if cxx is None:
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)

        if maxVal >= VALID_BRIGHTNESS_SMALL:
            cxx, cyy = maxLoc
            method = "MAXPIX"

    # -------------------------
    # Visualization
    # -------------------------
    if cxx is not None:
        color = (0, 0, 255) if method == "BLOB" else (255, 0, 0)
        cv2.circle(frame, (cxx, cyy), 1, color, 1)
        cv2.putText(
            frame,
            f"{method} ({cxx},{cyy})",
            (cxx + 6, cyy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )

    cv2.imshow("IR LED Tracking - Hybrid", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()