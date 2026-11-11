"""웹캠 열기 + MediaPipe 손 랜드마커 래퍼."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "assets" / "hand_landmarker.task"


def open_camera(index: int = 0, width: int = 1280, height: int = 720, fps: int = 30, mjpg: bool = True):
    """웹캠을 열고 (cap, 실제 너비, 실제 높이, 실제 FPS)를 돌려준다.

    ZIO-C960은 USB 2.0이라 큰 해상도에서 30 FPS를 내려면 MJPG(압축) 전송이 필요하다.
    MJPG는 해상도 설정보다 먼저 요청해야 적용되는 웹캠이 많다.
    """
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠 {index}번을 열 수 없습니다.")
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f = cap.get(cv2.CAP_PROP_FPS)
    return cap, w, h, f


class HandDetector:
    """프레임(BGR) → (랜드마크 (21,3) 또는 None, 왼손/오른손, 신뢰도). VIDEO 모드라 시각이 계속 증가해야 한다."""

    def __init__(self, model_path: Path = MODEL_PATH, min_detection_conf: float = 0.5):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path} (README의 다운로드 명령 참고)")
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_conf,
        )
        self._lm = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr, t_seconds: float):
        ts = max(int(t_seconds * 1000), self._last_ts + 1)
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not result.hand_landmarks:
            return None, "", 0.0
        pts = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]])
        hand = result.handedness[0][0]
        return pts, hand.category_name, float(hand.score)

    def close(self):
        self._lm.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_hand(frame, landmarks, color=(0, 200, 0)):
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
