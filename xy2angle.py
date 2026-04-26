import numpy as np

# ── 카메라 내부 파라미터 ──────────────────────────────────────
fx: float = 655.0
fy: float = 655.0
cx: float = 325.0
cy: float = 250.0
# ─────────────────────────────────────────────────────────────


def pixel_to_angles(
    u: float,
    v: float,
    cx: float = cx,
    cy: float = cy,
    fx: float = fx,
    fy: float = fy,
):
    """
    픽셀 좌표 (u, v)와 주점 (cx, cy) 사이의 각도를 반환한다.

    Parameters
    ----------
    u, v : 픽셀 좌표
    cx, cy : 주점 (principal point)
    fx, fy : 초점거리 (픽셀 단위)

    Returns
    -------
    yaw   : 수평 편각 [deg]  — 오른쪽(+), 왼쪽(−)
    pitch : 수직 편각 [deg]  — 위(+), 아래(−)
    """
    yaw   = np.degrees(np.arctan2(u - cx, fx))
    pitch = np.degrees(np.arctan2(cy - v, fy))
    return yaw, pitch