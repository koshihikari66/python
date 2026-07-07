"""
angle_logger.py
────────────────────────────────────────────────────────────
검출된 객체와 (cx, cy) 사이의 "단일 각도"(yaw/pitch로 분리하지 않은,
광축 기준 편각의 크기)를 시간과 함께 외부 CSV 파일에 기록하는 모듈.

라즈베리파이 자원 소모를 줄이기 위한 설계:
  1. 매 프레임 디스크에 쓰지 않는다.
     내부 큐에 (시간, 각도)만 넣고 즉시 반환(non-blocking) → 메인 루프
     (카메라 캡처 / 서보 제어) 지연 없음.
  2. 실제 파일 I/O는 별도 스레드에서 처리하며, N개가 쌓이거나 T초가
     지나면 한 번에 모아서(batch) write → SD카드 쓰기 syscall 횟수 최소화.
  3. numpy 등 무거운 라이브러리 없이 math + threading + queue만 사용
     (이미 메인 프로세스에 로드된 것들이라 추가 메모리 부담 거의 없음).
  4. CSV 포맷(time_s, angle_deg)이라 이후 pandas/matplotlib로 바로 로드 가능.

nnewredc.py 사용 예
────────────────────────────────────────────────────────────
    from angle_logger import AngleLogger, combined_angle

    logger = AngleLogger("angle_log.csv")   # main() 시작 부분에서 1회 생성

    ...
    if px is not None:
        angle = combined_angle(px, py)      # (cx,cy) 기준 단일 각도 [deg]
        logger.log(angle)                   # 큐에 넣기만 함, 거의 0비용
    ...

    finally:
        logger.close()                      # 프로그램 종료 시 반드시 호출
        servo.stop()
        cap.release()
"""

import math
import queue
import threading
import time

import xy2angle


# ── 각도 계산 ────────────────────────────────────────────────
def combined_angle(
    u: float,
    v: float,
    cx: float = None,
    cy: float = None,
    fx: float = None,
    fy: float = None,
) -> float:
    """
    주점 (cx, cy)과 픽셀 (u, v) 사이의 "단일" 각도 [deg].

    yaw/pitch로 분리하지 않고, 광축과 객체 방향 사이의 순수한
    각도 크기(광축으로부터 얼마나 벗어났는가)만 반환한다.
    fx, fy가 다를 수 있으므로 정규화된 카메라 좌표계에서 계산.

    인자를 생략하면 xy2angle.py에 정의된 현재 카메라 파라미터를 사용한다.
    """
    if cx is None:
        cx = xy2angle.getcx()
    if cy is None:
        cy = xy2angle.getcy()
    if fx is None:
        fx = xy2angle.getfx()
    if fy is None:
        fy = fx  # xy2angle 모듈이 fy는 별도로 노출하지 않으므로 fx로 대체

    xn = (u - cx) / fx
    yn = (v - cy) / fy
    return math.degrees(math.atan(math.hypot(xn, yn)))


# ── 비동기 버퍼링 로거 ──────────────────────────────────────
class AngleLogger:
    """
    (시간, 각도) 값을 큐에 쌓아두고, 별도 스레드가 일정 주기로
    묶어서 디스크에 기록하는 저자원 로거.
    """

    def __init__(
        self,
        filepath: str = "angle_log.csv",
        flush_interval: float = 2.0,   # 몇 초마다 디스크에 쓸지
        flush_count: int = 60,         # 몇 개 쌓이면 디스크에 쓸지
        t0: float = None,              # 시간 축 기준점 (None → 첫 log() 시각)
    ):
        self.filepath       = filepath
        self.flush_interval = flush_interval
        self.flush_count    = flush_count
        self._t0            = t0

        self._queue = queue.Queue()
        self._stop  = threading.Event()

        # 헤더 작성 (덮어쓰기 시작)
        with open(self.filepath, "w") as f:
            f.write("time_s,angle_deg\n")

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def log(self, angle_deg: float, t: float = None):
        """
        메인(실시간) 루프에서 호출하는 함수.
        큐에 값만 넣고 바로 반환하므로 프레임 처리 속도에 영향 없음.
        """
        now = t if t is not None else time.time()
        if self._t0 is None:
            self._t0 = now
        self._queue.put_nowait((now - self._t0, angle_deg))

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
                    f.write("".join(f"{t:.4f},{a:.4f}\n" for t, a in buf))
                    f.flush()
                    buf.clear()
                    last_flush = now

            # 종료 직전 남은 버퍼 정리
            if buf:
                f.write("".join(f"{t:.4f},{a:.4f}\n" for t, a in buf))
                f.flush()

    def close(self, timeout: float = 5.0):
        """
        프로그램 종료 시 반드시 호출.
        남은 큐 내용을 모두 디스크에 쓰고 스레드를 안전하게 종료한다.
        """
        self._stop.set()
        self._thread.join(timeout=timeout)
