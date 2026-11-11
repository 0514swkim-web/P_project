"""랜드마크 한 프레임 → /hand/state 한 개. 전처리·판별·상태 머신을 한데 묶는다."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import feature_vector, normalize_landmarks, palm_center
from .rules import NONE, RuleConfig, classify_normalized
from .state_machine import ClutchStateMachine, ControlConfig, Gripper, Mode


@dataclass
class HandState:
    """docs/hand_state_format.md 초안과 같은 필드 (ROS 메시지로 옮기기 전 파이썬 버전)."""

    stamp: float
    frame_index: int
    hand_detected: bool
    handedness: str
    detection_confidence: float
    position: tuple  # (x, y) 정규화 이미지 좌표
    control_mode: Mode
    hold_position: bool
    height_input: float
    gripper_state: Gripper
    gesture: str
    gesture_confidence: float


class GestureRecognizer:
    """판별 방식을 바꿔 끼울 수 있다: classifier=None이면 규칙 기반, 학습 모델을 주면 분류기."""

    def __init__(
        self,
        aspect: float,
        classifier=None,
        rule_cfg: RuleConfig | None = None,
        ctrl_cfg: ControlConfig | None = None,
        min_detection_conf: float = 0.5,
        min_gesture_conf: float = 0.3,
    ):
        self.aspect = aspect
        self.classifier = classifier
        self.rule_cfg = rule_cfg or RuleConfig()
        self.machine = ClutchStateMachine(ctrl_cfg)
        self.min_detection_conf = min_detection_conf
        self.min_gesture_conf = min_gesture_conf
        self._frame = 0
        self._last_pos = (0.0, 0.0)

    def reset(self):
        self.machine.reset()
        self._frame = 0

    def _classify(self, landmarks, handedness):
        if self.classifier is None:
            r = classify_normalized(normalize_landmarks(landmarks, handedness, self.aspect), self.rule_cfg)
            return r.gesture, r.confidence
        x = feature_vector(landmarks, handedness, self.aspect)[None, :]
        proba = self.classifier.predict_proba(x)[0]
        i = int(np.argmax(proba))
        return str(self.classifier.classes_[i]), float(proba[i])

    def process(self, t: float, landmarks, handedness: str = "", detection_conf: float = 0.0) -> HandState:
        """landmarks가 None이면 손 없음."""
        self._frame += 1
        detected = landmarks is not None
        gesture, gconf = NONE, 0.0
        pos = self._last_pos
        if detected:
            pos = tuple(float(v) for v in palm_center(landmarks))
            self._last_pos = pos
            gesture, gconf = self._classify(np.asarray(landmarks), handedness)
            if gconf < self.min_gesture_conf:
                gesture = NONE  # 애매하면 "모름" 취급 (판단이 애매하면 정지)

        hand_ok = detected and detection_conf >= self.min_detection_conf
        out = self.machine.update(t, hand_ok, gesture, pos[1])
        return HandState(
            stamp=t,
            frame_index=self._frame,
            hand_detected=detected,
            handedness=handedness,
            detection_confidence=detection_conf,
            position=pos,
            control_mode=out.control_mode,
            hold_position=out.hold_position,
            height_input=out.height_input,
            gripper_state=out.gripper_state,
            gesture=gesture,
            gesture_confidence=gconf,
        )
