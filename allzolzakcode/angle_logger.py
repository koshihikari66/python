"""
angle_logger.py
────────────────────────────────────────────────────────────
검출된 객체의 픽셀 좌표 (u, v)를 기준으로
  - 광축 기준 단일 편각 angle
  - 수평 방향 yaw
  - 수직 방향 pitch
를 시간과 함께 CSV 파일에 기록하는 모듈.

CSV 컬럼:
    time_s, angle_deg, yaw_deg, pitch_deg

권장 사용법
────────────────────────────────────────────────────────────
    from angle_logger_pitch_yaw import AngleLogger

    logger = AngleLogger("angle_log.csv")

    ...
    if px is not None:
        logger.log_pixel(px, py)
    ...

    finally:
        logger.close()

기존 코드 호환:
    logger.log(angle)
도 그대로 사용할 수 있지만, 이 경우 픽셀 좌표가 없으므로 yaw/pitch는 NaN으로 기록된다.
"""

import math
import queue
import threading
import time

import xy2angle


# ── 카메라 파라미터 정리 ────────────────────────────────────
def _camera_params(
    cx: float = None,
    cy: float = None,
    fx: float = None,
    fy: float = None,
):
    if cx is None:
        cx = xy2angle.getcx()
    if cy is None:
        cy = xy2angle.getcy()
    if fx is None:
        fx = xy2angle.getfx()
    if fy is None:
        # xy2angle 모듈이 fy를 별도로 노출하지 않는 경우 fx 사용
        fy = fx
    return cx, cy, fx, fy


# ── 각도 계산 ────────────────────────────────────────────────
def yaw_pitch_angles(
    u: float,
    v: float,
    cx: float = None,
    cy: float = None,
    fx: float = None,
    fy: float = None,
):
    """
    픽셀 좌표 (u, v)의 yaw, pitch [deg]를 반환한다.

    부호 기준:
      yaw   > 0 : 영상 중심보다 오른쪽
      yaw   < 0 : 영상 중심보다 왼쪽
      pitch > 0 : 영상 중심보다 아래쪽
      pitch < 0 : 영상 중심보다 위쪽

    카메라 좌표의 수직축을 위쪽(+)으로 쓰고 싶다면
    pitch 계산식에 -(v - cy)를 사용하면 된다.
    """
    cx, cy, fx, fy = _camera_params(cx, cy, fx, fy)

    yaw_deg = math.degrees(math.atan((u - cx) / fx))
    pitch_deg = math.degrees(math.atan((v - cy) / fy))
    return yaw_deg, pitch_deg


def combined_angle(
    u: float,
    v: float,
    cx: float = None,
    cy: float = None,
    fx: float = None,
    fy: float = None,
) -> float:
    """
    주점 (cx, cy)과 픽셀 (u, v) 사이의 단일 편각 [deg].
    yaw/pitch의 부호와 무관하게 광축에서 얼마나 벗어났는지를 나타낸다.
    """
    cx, cy, fx, fy = _camera_params(cx, cy, fx, fy)

    xn = (u - cx) / fx
    yn = (v - cy) / fy
    return math.degrees(math.atan(math.hypot(xn, yn)))


def all_angles(
    u: float,
    v: float,
    cx: float = None,
    cy: float = None,
    fx: float = None,
    fy: float = None,
):
    """(angle_deg, yaw_deg, pitch_deg)를 한 번에 반환한다."""
    cx, cy, fx, fy = _camera_params(cx, cy, fx, fy)

    xn = (u - cx) / fx
    yn = (v - cy) / fy

    angle_deg = math.degrees(math.atan(math.hypot(xn, yn)))
    yaw_deg = math.degrees(math.atan(xn))
    pitch_deg = math.degrees(math.atan(yn))
    return angle_deg, yaw_deg, pitch_deg


# ── 비동기 버퍼링 로거 ──────────────────────────────────────
class AngleLogger:
    """
    (시간, 단일각도, yaw, pitch)를 큐에 쌓아두고,
    별도 스레드가 일정 주기로 묶어서 CSV에 기록하는 저자원 로거.
    """

    def __init__(
        self,
        filepath: str = "angle_log.csv",
        flush_interval: float = 2.0,
        flush_count: int = 60,
        t0: float = None,
    ):
        self.filepath = filepath
        self.flush_interval = flush_interval
        self.flush_count = flush_count
        self._t0 = t0

        self._queue = queue.Queue()
        self._stop = threading.Event()

        # 헤더 작성 (덮어쓰기 시작)
        with open(self.filepath, "w") as f:
            f.write("time_s,angle_deg,yaw_deg,pitch_deg\n")

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _put(self, angle_deg: float, yaw_deg: float, pitch_deg: float, t: float = None):
        now = t if t is not None else time.time()
        if self._t0 is None:
            self._t0 = now

        self._queue.put_nowait(
            (now - self._t0, angle_deg, yaw_deg, pitch_deg)
        )

    def log(
        self,
        angle_deg: float,
        yaw_deg: float = None,
        pitch_deg: float = None,
        t: float = None,
    ):
        """
        이미 계산된 각도를 기록한다.

        예:
            logger.log(angle, yaw, pitch)

        기존 코드처럼 logger.log(angle)만 호출하면
        yaw/pitch는 계산할 정보가 없으므로 NaN으로 기록한다.
        """
        if yaw_deg is None:
            yaw_deg = math.nan
        if pitch_deg is None:
            pitch_deg = math.nan

        self._put(angle_deg, yaw_deg, pitch_deg, t=t)

    def log_pixel(
        self,
        u: float,
        v: float,
        t: float = None,
        cx: float = None,
        cy: float = None,
        fx: float = None,
        fy: float = None,
    ):
        """
        픽셀 좌표 (u, v)를 받아 angle/yaw/pitch를 계산하고 바로 기록한다.

        메인 추적 루프에서는 이 함수를 쓰는 것이 가장 간단하다.
        """
        angle_deg, yaw_deg, pitch_deg = all_angles(
            u, v, cx=cx, cy=cy, fx=fx, fy=fy
        )
        self._put(angle_deg, yaw_deg, pitch_deg, t=t)
        return angle_deg, yaw_deg, pitch_deg

    def _writer_loop(self):
        buf = []
        last_flush = time.time()

        with open(self.filepath, "a", buffering=8192) as f:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.5)
                    buf.append(item)
                except queue.Empty:
                    pass

                now = time.time()
                if buf and (
                    len(buf) >= self.flush_count
                    or (now - last_flush) >= self.flush_interval
                ):
                    f.write(
                        "".join(
                            f"{t:.4f},{a:.4f},{y:.4f},{p:.4f}\n"
                            for t, a, y, p in buf
                        )
                    )
                    f.flush()
                    buf.clear()
                    last_flush = now

            # 종료 직전 남은 버퍼 정리
            if buf:
                f.write(
                    "".join(
                        f"{t:.4f},{a:.4f},{y:.4f},{p:.4f}\n"
                        for t, a, y, p in buf
                    )
                )
                f.flush()

    def close(self, timeout: float = 5.0):
        """남은 큐 내용을 모두 디스크에 쓰고 스레드를 안전하게 종료한다."""
        self._stop.set()
        self._thread.join(timeout=timeout)