import cv2
import numpy as np
from collections import deque

# ======================
# 설정
# ======================
FPS = 30
HISTORY_SEC = 1.0
HISTORY_LEN = int(FPS * HISTORY_SEC)

BRIGHT_THRESH = 200
MIN_AREA = 3
MAX_AREA = 100

# duty 조건
MIN_ON_RATIO = 0.35
MAX_ON_RATIO = 0.65
MIN_TOGGLES = 20

# ======================
# 카메라
# ======================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, FPS)

tracks = []  # {'pt':(x,y), 'state_hist':deque}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 강한 threshold (IR 밝은 점)
    _, th = cv2.threshold(gray, BRIGHT_THRESH, 255, cv2.THRESH_BINARY)
    th = cv2.medianBlur(th, 5)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2
            detections.append((cx, cy))

    # tracking
    for (cx, cy) in detections:
        matched = False
        for t in tracks:
            px, py = t['pt']
            if np.hypot(cx - px, cy - py) < 15:
                t['pt'] = (cx, cy)
                state = 1 if gray[cy, cx] > BRIGHT_THRESH else 0
                t['state_hist'].append(state)
                matched = True
                break

        if not matched:
            tracks.append({
                'pt': (cx, cy),
                'state_hist': deque(maxlen=HISTORY_LEN)
            })

    # 판단
    for t in tracks:
        hist = list(t['state_hist'])
        if len(hist) < HISTORY_LEN:
            continue

        on_count = sum(hist)
        on_ratio = on_count / len(hist)

        toggles = sum(
            1 for i in range(1, len(hist)) if hist[i] != hist[i-1]
        )

        if (MIN_ON_RATIO < on_ratio < MAX_ON_RATIO) and toggles >= MIN_TOGGLES:
            x, y = t['pt']
            cv2.circle(frame, (x, y), 8, (0, 255, 0), 2)
            cv2.putText(frame, "15Hz IR LED",
                        (x+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 1)

    cv2.imshow("Duty-based IR LED Detect", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()