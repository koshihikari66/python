import cv2
import numpy as np

# ── 파라미터 ──────────────────────────────────────────────
CAM_ID       = 0
WIDTH        = 640
HEIGHT       = 480

# HSV 빨간색 범위 (빨강은 0° 근처와 160°~ 두 구간)
H_LOW1,  S_LOW1,  V_LOW1  =   0, 60, 100
H_HIGH1, S_HIGH1, V_HIGH1 =  10, 255, 255
H_LOW2,  S_LOW2,  V_LOW2  = 160, 60, 100
H_HIGH2, S_HIGH2, V_HIGH2 = 180, 255, 255

# LED 검출 조건
MIN_AREA   = 1      # 최소 픽셀 수 (LED 1개 = 1~수십 픽셀)
MAX_AREA   = 500    # 너무 넓으면 노이즈로 간주

# 모폴로지 커널
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# ────────────────────────────────────────────────────────


def subpixel_centroid(gray: np.ndarray, mask: np.ndarray):
    """
    마스크 영역 내 밝기 값을 가중치로 사용해 무게중심(sub-pixel) 좌표를 계산.
    반환: (cx, cy) float  /  검출 실패 시 None
    """
    roi = gray.astype(np.float64)
    roi[mask == 0] = 0.0

    total = roi.sum()
    if total == 0:
        return None

    # meshgrid로 한번에 계산
    ys, xs = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    cx = (xs * roi).sum() / total
    cy = (ys * roi).sum() / total
    return cx, cy


def detect_red_led(frame: np.ndarray):
    """
    frame BGR → 빨간 LED 후보 목록 반환
    각 항목: {'centroid': (cx, cy), 'area': int, 'bbox': (x,y,w,h)}
    """
    # 노이즈 제거용 약한 블러
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray    = cv2.cvtColor(frame,   cv2.COLOR_BGR2GRAY)

    # 빨간색 두 구간 마스크 합산
    mask1 = cv2.inRange(hsv,
                        (H_LOW1, S_LOW1, V_LOW1),
                        (H_HIGH1, S_HIGH1, V_HIGH1))
    mask2 = cv2.inRange(hsv,
                        (H_LOW2, S_LOW2, V_LOW2),
                        (H_HIGH2, S_HIGH2, V_HIGH2))
    mask  = cv2.bitwise_or(mask1, mask2)

    # 모폴로지: 작은 구멍 채우기 + 노이즈 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL, iterations=1)

    # 연결 컴포넌트 분석
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    results = []
    for i in range(1, num_labels):          # 0 = 배경
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


def draw_results(frame: np.ndarray, detections: list) -> np.ndarray:
    vis = frame.copy()
    for det in detections:
        cx, cy = det['centroid']
        x, y, w, h = det['bbox']

        # 바운딩 박스
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # 서브픽셀 중심 십자선 (소수점 좌표 → LineIterator 대신 정수 근사 표시)
        icx, icy = int(round(cx)), int(round(cy))
        cv2.drawMarker(vis, (icx, icy), (0, 255, 255),
                       cv2.MARKER_CROSS, markerSize=15, thickness=1,
                       line_type=cv2.LINE_AA)

        # 좌표 텍스트 (소수점 2자리)
        label = f"({cx:.2f}, {cy:.2f})  A={det['area']}"
        cv2.putText(vis, label,
                    (x, y - 6 if y > 16 else y + h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                    cv2.LINE_AA)

    # 상태 표시
    status = f"Detected: {len(detections)}"
    cv2.putText(vis, status, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("=" * 50)
    print("  빨간 LED 서브픽셀 검출기")
    print("  q / ESC : 종료")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break

        detections, mask = detect_red_led(frame)
        vis = draw_results(frame, detections)

        # 콘솔 출력
        for det in detections:
            cx, cy = det['centroid']
            print(f"[LED] centroid=({cx:.4f}, {cy:.4f})  area={det['area']}px")

        # 마스크를 컬러로 변환해 나란히 표시
        mask_bgr  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined  = np.hstack([vis, mask_bgr])

        cv2.imshow("Red LED Detector  |  [original]  [mask]", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()