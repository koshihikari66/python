import cv2
import numpy as np
import glob
import os

# 설정
SAVE_DIR = "calib_images"
CHECKERBOARD = (8, 6)       # 내부 코너 수 (촬영 설정과 동일하게)
SQUARE_SIZE = 25.0          # 체커보드 한 칸 실제 크기 (mm) ← 본인 체커보드에 맞게 수정
IMAGE_SIZE = (640, 480)

# 3D 기준점 생성
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D 실세계 좌표
imgpoints = []  # 2D 이미지 좌표

images = sorted(glob.glob(os.path.join(SAVE_DIR, "*.jpg")))
print(f"총 {len(images)}장 발견\n")

valid_count = 0
fail_count = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        valid_count += 1
        print(f"  ✔ {os.path.basename(fname)}")
    else:
        fail_count += 1
        print(f"  ✘ {os.path.basename(fname)} (코너 미감지 - 제외)")

print(f"\n유효: {valid_count}장 / 실패: {fail_count}장")

if valid_count < 10:
    print("⚠ 유효 이미지가 너무 적습니다. 최소 10장 이상 권장합니다.")
    exit()

# ── 캘리브레이션 실행 ──────────────────────────────────────────────
print("\n캘리브레이션 계산 중...")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, IMAGE_SIZE, None, None)

# ── Reprojection Error 계산 ────────────────────────────────────────
errors = []
for i in range(len(objpoints)):
    projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
    errors.append(err)

mean_error = np.mean(errors)

# ── 결과 출력 ──────────────────────────────────────────────────────
print("\n" + "="*50)
print("         캘리브레이션 결과")
print("="*50)
print(f"\n📊 Reprojection Error: {mean_error:.4f} px", end="  ")
if mean_error < 0.5:
    print("✅ 우수")
elif mean_error < 1.0:
    print("⚠ 보통 (일부 사진 제거 후 재시도 권장)")
else:
    print("❌ 불량 (사진 재촬영 필요)")

print(f"\n📷 Camera Matrix (K):\n{K}")
print(f"\n🔧 Distortion Coefficients:\n{dist.ravel()}")

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]
print(f"\n  fx={fx:.2f}, fy={fy:.2f}")
print(f"  cx={cx:.2f}, cy={cy:.2f}")

# ── 결과 저장 ──────────────────────────────────────────────────────
np.savez("camera_calib.npz",
         K=K, dist=dist,
         reprojection_error=mean_error,
         image_size=IMAGE_SIZE)
print("\n💾 camera_calib.npz 저장 완료")

# ── 왜곡 보정 미리보기 ─────────────────────────────────────────────
print("\n[미리보기] 아무 키나 누르면 다음 이미지 / Q 종료")

for fname in images[:5]:  # 처음 5장만 미리보기
    img = cv2.imread(fname)
    undistorted = cv2.undistort(img, K, dist)

    combined = np.hstack([img, undistorted])
    cv2.putText(combined, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(combined, "Undistorted", (650, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Before / After", combined)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# ── 나중에 불러쓰는 방법 ───────────────────────────────────────────
print("""
──────────────────────────────────────
나중에 결과 불러오기:

  data = np.load('camera_calib.npz')
  K    = data['K']
  dist = data['dist']

  undistorted = cv2.undistort(frame, K, dist)
──────────────────────────────────────
""")