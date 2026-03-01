import cv2
import numpy as np

fx = 655.0
fy = 655.0
cx = 325.0
cy = 250.0
#dist = np.array([0.02076, -0.01249, 0.00532, 0.00447, 0.06789])
prev_time = 0

K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float64)

selected_point = None
CENTER = (int(cx), int(cy))

def pixel_to_ray(u, v):
    pt = np.array([[[u, v]]], dtype=np.float64)
#    pt_undist = cv2.undistortPoints(pt, K, dist, P=K)
#    u2, v2 = pt_undist[0, 0]
    ray = np.array([(u - cx) / fx,
                    (v - cy) / fy,
                    1.0])
    return ray / np.linalg.norm(ray)  # 단위 벡터

def angle_between(ray1, ray2):
    r1_xz = np.array([ray1[0], 0, ray1[2]])
    r2_xz = np.array([ray2[0], 0, ray2[2]])
    r1_xz /= np.linalg.norm(r1_xz)
    r2_xz /= np.linalg.norm(r2_xz)
    cos_yaw = np.clip(np.dot(r1_xz, r2_xz), -1.0, 1.0)
    yaw = np.degrees(np.arccos(cos_yaw))
    yaw *= np.sign(ray2[0] - ray1[0])  #오른쪽 +

    r1_yz = np.array([0, ray1[1], ray1[2]])
    r2_yz = np.array([0, ray2[1], ray2[2]])
    r1_yz /= np.linalg.norm(r1_yz)
    r2_yz /= np.linalg.norm(r2_yz)
    cos_pitch = np.clip(np.dot(r1_yz, r2_yz), -1.0, 1.0)
    pitch = np.degrees(np.arccos(cos_pitch))
    pitch *= np.sign(ray1[1] - ray2[1])  #위쪽 +

    return yaw, pitch

def mouse_callback(event, x, y, flags, param):
    global selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)

cap = cv2.VideoCapture(cv2.CAP_DSHOW+0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

ray_center = pixel_to_ray(*CENTER)
cv2.namedWindow("Angle Viewer")
cv2.setMouseCallback("Angle Viewer", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cropped = frame[120:600, 320:960]
    display = cropped.copy()

    # FPS
    curr_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (curr_time - prev_time)
    prev_time = curr_time

    cv2.drawMarker(display, CENTER, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(display, "Center", (CENTER[0]+8, CENTER[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(display, f"FPS: {fps:.1f}", (540, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if selected_point is not None:
        px, py = selected_point
        ray_p = pixel_to_ray(px, py)

        yaw, pitch = angle_between(ray_center, ray_p)

        cv2.line(display, CENTER, (px, py), (0, 200, 255), 1)
        cv2.circle(display, (px, py), 1, (0, 100, 255), 1)

        texts = [
            f"Selected : ({px}, {py})",
            f"Yaw      : {yaw:+.2f} deg",
            f"Pitch    : {pitch:+.2f} deg",
        ]
        cv2.rectangle(display, (0, 424), (420, 480), (0, 0, 0), -1)
        for i, t in enumerate(texts):
            cv2.putText(display, t, (8, 438 + i * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 180), 1)

    cv2.imshow("Angle Viewer", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()