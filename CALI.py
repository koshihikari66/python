import cv2
import os
import numpy as np

# 설정
SAVE_DIR = "calib_images"
os.makedirs(SAVE_DIR, exist_ok=True)

CHECKERBOARD = (8, 6)  # 내부 코너 수
cap = cv2.VideoCapture(cv2.CAP_DSHOW+0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -3)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("=== 캘리브레이션 이미지 촬영 ===")
print("  [SPACE] : 사진 저장")
print("  [Q]     : 종료")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1280x720 → 640x480 센터 크롭
    h, w = frame.shape[:2]  # 720, 1280
    x1 = (w - 640) // 2     # 320
    y1 = (h - 480) // 2     # 120
    cropped = frame[y1:y1+480, x1:x1+640]

    display = cropped.copy()

    # 체커보드 코너 실시간 감지
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                        cv2.CALIB_CB_FAST_CHECK)

    if found:
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)
        status_color = (0, 255, 0)
        status_text = f"감지됨! [SPACE]로 저장 ({count}장)"
    else:
        status_color = (0, 0, 255)
        status_text = f"미감지 ({count}장 저장됨)"

    # UI 오버레이
    cv2.rectangle(display, (0, 0), (640, 30), (0, 0, 0), -1)
    cv2.putText(display, status_text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # 크롭 영역 표시 (참고용 테두리)
    cv2.rectangle(display, (0, 0), (639, 479), (255, 255, 0), 1)

    cv2.imshow("Calibration Capture (640x480 cropped)", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        if found:
            fname = os.path.join(SAVE_DIR, f"calib_{count:03d}.jpg")
            cv2.imwrite(fname, cropped)  # 크롭된 원본 저장 (오버레이 없이)
            count += 1
            print(f"[저장] {fname}")
        else:
            print("[경고] 체커보드가 감지되지 않아 저장하지 않았습니다.")

cap.release()
cv2.destroyAllWindows()
print(f"\n총 {count}장 저장 완료 → '{SAVE_DIR}/' 폴더 확인")