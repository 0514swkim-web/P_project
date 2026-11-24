"""실시간 처리 한 프레임 + 화면 표시. scripts/live_demo.py와 (나중에) ROS 2 노드가 같이 쓴다.

카메라 루프와 분리해 두었기 때문에, 검출기 자리에 가짜 객체를 넣으면 웹캠 없이 테스트할 수 있다.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .features import normalize_landmarks
from .pipeline import GestureRecognizer, HandState
from .rules import FingerMetrics, RuleConfig, finger_metrics, finger_states
from .state_machine import ClutchStateMachine, Mode, roles_for_clutch

CLUTCH_CHOICES = ("call", "rock", "ok", "palm")  # palm = 베이스라인(단순 활성화)


def free_output_paths(*paths):
    """녹화·로그 파일이 이미 있으면 덮어쓰지 않도록 같은 번호(_2, _3 ...)를 붙인 경로를 돌려준다.

    영상과 CSV가 짝이 맞게 모든 경로에 같은 번호를 붙인다. None은 그대로 둔다.
    (11/24 같은 이름으로 여러 번 실행해 A·B 녹화가 덮어써진 일이 있었음)
    """
    from pathlib import Path

    given = [Path(p) if p else None for p in paths]
    n = 1
    while True:
        cand = [p if (p is None or n == 1) else p.with_name(f"{p.stem}_{n}{p.suffix}") for p in given]
        if not any(p is not None and p.exists() for p in cand):
            return [str(p) if p is not None else None for p in cand]
        n += 1

# BGR
MODE_COLORS = {
    Mode.IDLE: (180, 180, 180),
    Mode.PLANAR: (80, 220, 80),
    Mode.HEIGHT: (0, 200, 255),
    Mode.PAUSED: (60, 60, 255),
}


@dataclass
class FrameResult:
    state: HandState
    landmarks: np.ndarray | None
    fingers: tuple | None  # 규칙 기준 손가락 펴짐 (엄지~소지)
    metrics: FingerMetrics | None
    clutch_progress: float
    detect_ms: float  # MediaPipe 손 인식 시간
    process_ms: float  # 판별 + 상태 머신 시간 (계획서 목표: 판별 5ms 미만)


class LiveSession:
    """detector: .detect(frame_bgr, t) → (landmarks|None, handedness, score) 를 가진 객체."""

    def __init__(self, detector, recognizer: GestureRecognizer, clutch: str = "call"):
        self.detector = detector
        self.recognizer = recognizer
        self.clutch = clutch
        self.set_clutch(clutch)

    def set_clutch(self, clutch: str):
        """클러치 후보를 바꾼다 (상태 머신을 새로 만들어 초기화된다)."""
        cfg = dataclasses.replace(self.recognizer.machine.cfg, roles=roles_for_clutch(clutch))
        self.recognizer.machine = ClutchStateMachine(cfg)
        self.clutch = clutch

    def next_clutch(self) -> str:
        i = CLUTCH_CHOICES.index(self.clutch) if self.clutch in CLUTCH_CHOICES else -1
        self.set_clutch(CLUTCH_CHOICES[(i + 1) % len(CLUTCH_CHOICES)])
        return self.clutch

    def reset(self):
        self.recognizer.reset()

    def step(self, frame_bgr, t: float) -> FrameResult:
        t0 = time.perf_counter()
        lm, hand, score = self.detector.detect(frame_bgr, t)
        t1 = time.perf_counter()
        state = self.recognizer.process(t, lm, hand, score)
        t2 = time.perf_counter()

        fingers = metrics = None
        if lm is not None:
            norm = normalize_landmarks(np.asarray(lm), hand, self.recognizer.aspect)
            metrics = finger_metrics(norm)
            fingers, _ = finger_states(norm, self.recognizer.rule_cfg)
        return FrameResult(
            state=state,
            landmarks=lm,
            fingers=fingers,
            metrics=metrics,
            clutch_progress=self.recognizer.machine.clutch_progress(t),
            detect_ms=(t1 - t0) * 1000,
            process_ms=(t2 - t1) * 1000,
        )


def state_row(r: FrameResult, clutch: str) -> dict:
    """프레임별 로그 CSV 한 줄 (/hand/state 필드 + 시간 측정)."""
    s = r.state
    return {
        "stamp": round(s.stamp, 4),
        "frame_index": s.frame_index,
        "hand_detected": int(s.hand_detected),
        "handedness": s.handedness,
        "detection_confidence": round(s.detection_confidence, 4),
        "x": round(s.position[0], 4),
        "y": round(s.position[1], 4),
        "control_mode": s.control_mode.name,
        "hold_position": int(s.hold_position),
        "height_input": round(s.height_input, 4),
        "gripper_state": s.gripper_state.name,
        "gesture": s.gesture,
        "gesture_confidence": round(s.gesture_confidence, 4),
        "clutch": clutch,
        "detect_ms": round(r.detect_ms, 3),
        "process_ms": round(r.process_ms, 3),
        # 튜닝용 원시 값 (손이 없으면 빈칸). 녹화 영상과 같이 남기면 화면을 읽지 않고 바로 분석할 수 있다.
        **_metric_cols(r),
    }


def _metric_cols(r: FrameResult) -> dict:
    m = r.metrics
    if m is None:
        return dict.fromkeys(("fingers", "thumb_ang", "thumb_dist", "index_ang", "middle_ang",
                              "ring_ang", "pinky_ang", "ok_dist"), "")
    return {
        "fingers": "".join(map(str, r.fingers)),
        "thumb_ang": round(m.thumb_angle, 1),
        "thumb_dist": round(m.thumb_dist, 3),
        **{f"{n}_ang": round(a, 1) for n, a in zip(("index", "middle", "ring", "pinky"), m.finger_angles)},
        "ok_dist": round(m.ok_dist, 3),
    }


# ---------------- 화면 표시 ----------------
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _box(frame, x, y, w, h, alpha=0.55):
    """반투명 검은 상자 (글자가 배경에 묻히지 않게)."""
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    roi = frame[y:y2, x:x2]
    roi[:] = (roi * (1 - alpha)).astype(frame.dtype)


def _bar(frame, x, y, w, label, value, vmin, vmax, threshold, on):
    """값 막대 + 임계값 선. on이면 초록(=임계값 넘음)."""
    h = 12
    frac = float(np.clip((value - vmin) / (vmax - vmin), 0, 1))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)
    cv2.rectangle(frame, (x, y), (x + int(w * frac), y + h), (80, 220, 80) if on else (150, 150, 150), -1)
    tx = x + int(w * np.clip((threshold - vmin) / (vmax - vmin), 0, 1))
    cv2.line(frame, (tx, y - 3), (tx, y + h + 3), (0, 0, 255), 2)
    cv2.putText(frame, f"{label} {value:.2f}" if vmax <= 2 else f"{label} {value:.0f}",
                (x + w + 8, y + h - 1), _FONT, 0.45, (255, 255, 255), 1)


def draw_overlay(frame, r: FrameResult, rule_cfg: RuleConfig, clutch: str, fps: float, detail: bool = True):
    """frame에 직접 그린다."""
    from .camera import draw_hand  # mediapipe import를 테스트에서 피하려고 지연 import

    s = r.state
    if r.landmarks is not None:
        draw_hand(frame, r.landmarks)
        h, w = frame.shape[:2]
        cv2.circle(frame, (int(s.position[0] * w), int(s.position[1] * h)), 8, (255, 0, 255), -1)

    # ---- 왼쪽 위: 상태 ----
    _box(frame, 0, 0, 430, 175)
    color = MODE_COLORS[s.control_mode]
    cv2.putText(frame, s.control_mode.name, (10, 38), _FONT, 1.1, color, 3)
    hold = "HOLD" if s.hold_position else "move"
    cv2.putText(frame, hold, (250, 38), _FONT, 0.8, (60, 60, 255) if s.hold_position else (80, 220, 80), 2)
    fingers = "".join(map(str, r.fingers)) if r.fingers else "-----"
    cv2.putText(frame, f"gesture: {s.gesture} ({s.gesture_confidence:.2f})  [{fingers}]",
                (10, 68), _FONT, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"gripper: {s.gripper_state.name}   height: {s.height_input:+.3f}",
                (10, 96), _FONT, 0.6, (255, 255, 255), 1)
    # 클러치 게이지
    cv2.putText(frame, f"clutch: {clutch}", (10, 124), _FONT, 0.6, (255, 255, 255), 1)
    gx, gw = 160, 250
    cv2.rectangle(frame, (gx, 110), (gx + gw, 126), (90, 90, 90), 1)
    cv2.rectangle(frame, (gx, 110), (gx + int(gw * r.clutch_progress), 126), (0, 200, 255), -1)
    cv2.putText(frame, f"{fps:4.1f} FPS | detect {r.detect_ms:4.1f}ms | judge {r.process_ms:4.2f}ms",
                (10, 158), _FONT, 0.5, (200, 200, 200), 1)

    # ---- 오른쪽: 손가락 원시 값 (튜닝용) ----
    if detail and r.metrics is not None:
        m, c = r.metrics, rule_cfg
        fx = frame.shape[1] - 330
        _box(frame, fx - 10, 0, 340, 200)
        cv2.putText(frame, "rule metrics (red = threshold)", (fx, 20), _FONT, 0.45, (255, 255, 255), 1)
        thumb_on = bool(r.fingers and r.fingers[0])
        _bar(frame, fx, 32, 180, "thumb ang", m.thumb_angle, 60, 180, c.thumb_extended_deg,
             m.thumb_angle > c.thumb_extended_deg)
        _bar(frame, fx, 54, 180, "thumb dist", m.thumb_dist, 0, 1.5, c.thumb_away_dist,
             m.thumb_dist > c.thumb_away_dist)
        for i, (name, a) in enumerate(zip(("index", "middle", "ring", "pinky"), m.finger_angles)):
            _bar(frame, fx, 82 + 22 * i, 180, name, a, 60, 180, c.finger_extended_deg, a > c.finger_extended_deg)
        _bar(frame, fx, 172, 180, "ok dist", m.ok_dist, 0, 1.0, c.ok_touch_dist, m.ok_dist < c.ok_touch_dist)
        if not thumb_on and m.thumb_angle > c.thumb_extended_deg:
            cv2.putText(frame, "thumb: angle ok, dist short", (fx, 196), _FONT, 0.4, (0, 200, 255), 1)

    cv2.putText(frame, "q quit | r reset | c clutch | d detail", (10, frame.shape[0] - 12),
                _FONT, 0.5, (255, 255, 255), 1)
    return frame
