"""규칙 기반 제스처 판별 (구현 5단계).

아이디어: 손가락 5개가 각각 펴졌는지(1)/접혔는지(0)를 관절 각도로 판정하고,
그 패턴을 제스처 표와 맞춰 본다. 학습이 필요 없어 분류기와 비교할 기준선(baseline)이 된다.

임계값: 엄지·OK는 11/21 첫 실측(야외 10 FPS, T01)으로 맞췄고, 검지~소지는 아직 초기값이다.
실내 30 FPS 데이터로 다시 확인한다. 근거는 docs/tuning_log.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import joint_angles, normalize_landmarks

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# 제스처별 손가락 패턴 (엄지, 검지, 중지, 약지, 소지). "ok"는 별도 판정.
# None = 상관없음: 판정에도, 신뢰도 계산에도 쓰지 않는다.
GESTURE_PATTERNS = {
    "palm": (1, 1, 1, 1, 1),
    "fist": (0, 0, 0, 0, 0),
    # V자는 엄지를 접는 사람도, 편 채로 하는 사람도 있다 (11/21 실측: 같은 사람도 녹화마다 달랐음)
    "peace": (None, 1, 1, 0, 0),
    "call": (1, 0, 0, 0, 1),
    "rock": (0, 1, 0, 0, 1),
    "like": (1, 0, 0, 0, 0),
    "one": (0, 1, 0, 0, 0),
}
NONE = "none"


@dataclass
class RuleConfig:
    finger_extended_deg: float = 150.0  # 검지~소지: PIP·DIP 각도 평균이 이보다 크면 펴짐
    thumb_extended_deg: float = 140.0  # 엄지: MCP·IP 각도 평균이 이보다 크고
    # 엄지끝(4)이 검지 뿌리(5)에서 이만큼 떨어져 있으면 펴짐 (손바닥 크기 = 1).
    # 11/21 실측(T01, 야외): 접힘 0.24~0.56 (V자·락·검지·주먹), 펴짐 0.97~1.45 (손바닥·샤카·엄지 척·OK) → 사이값 0.75
    thumb_away_dist: float = 0.75
    thumb_dist_margin: float = 0.3  # 신뢰도 계산: 엄지 거리가 임계값에서 이만큼 떨어지면 신뢰도 1
    # 엄지끝(4)~검지끝(8)이 이보다 가까우면 "맞닿음".
    # 11/21 실측: OK 0.46~0.61, 중지·약지·소지가 펴진 다른 손(손바닥) 1.33 이상 → 0.8
    ok_touch_dist: float = 0.8
    margin_deg: float = 30.0  # 신뢰도 계산: 임계값에서 이만큼 떨어지면 신뢰도 1


@dataclass
class RuleResult:
    gesture: str
    confidence: float
    fingers: tuple  # (엄지, 검지, 중지, 약지, 소지) 0/1


@dataclass
class FingerMetrics:
    """규칙 판정에 쓰는 원시 값. 임계값 튜닝 때 화면에 띄워 본다."""

    thumb_angle: float  # 엄지 MCP·IP 각도 평균 (도)
    thumb_dist: float  # 엄지끝(4)~검지 뿌리(5) 거리 (손바닥 크기 = 1)
    finger_angles: tuple  # 검지·중지·약지·소지의 PIP·DIP 각도 평균 (도)
    ok_dist: float  # 엄지끝(4)~검지끝(8) 거리


def finger_metrics(norm: np.ndarray) -> FingerMetrics:
    ang = joint_angles(norm)
    return FingerMetrics(
        thumb_angle=float((ang[1] + ang[2]) / 2),
        thumb_dist=float(np.linalg.norm(norm[4, :2] - norm[5, :2])),
        finger_angles=tuple(float((ang[3 * f + 1] + ang[3 * f + 2]) / 2) for f in range(1, 5)),
        ok_dist=float(np.linalg.norm(norm[4] - norm[8])),
    )


def thumb_state(angle: float, dist: float, cfg: RuleConfig) -> tuple[bool, float]:
    """엄지 펴짐 여부와 판정 여유(margin, 클수록 확실).

    엄지는 "각도도 펴졌고 AND 검지 뿌리에서 멀다"일 때만 펴짐이다.
    각 조건을 임계값 대비 부호 있는 여유로 바꾸면(+ 넘음, - 못 넘음):
      - 펴짐: 둘 다 + → 덜 넘은 쪽(min)이 여유
      - 접힘: 하나라도 - → 가장 크게 못 넘은 쪽(min)이 접힘의 근거
    즉 둘 다 |min|. 예전엔 접힘일 때도 통과한 조건(각도)의 여유까지 min에 넣어서,
    거리로 확실히 접힌 엄지(V자 등)도 신뢰도가 0.3 근처로 깎였다.
    """
    ang_m = (angle - cfg.thumb_extended_deg) / cfg.margin_deg
    dist_m = (dist - cfg.thumb_away_dist) / cfg.thumb_dist_margin
    worst = min(ang_m, dist_m)
    return worst > 0, abs(worst)


def finger_margins(norm: np.ndarray, cfg: RuleConfig) -> tuple[tuple, tuple]:
    """정규화 랜드마크 → (손가락 펴짐 패턴, 손가락별 판정 여유). 여유는 클수록 확실 (1 이상 = 충분)."""
    m = finger_metrics(norm)
    thumb_ext, thumb_margin = thumb_state(m.thumb_angle, m.thumb_dist, cfg)
    states, margins = [int(thumb_ext)], [thumb_margin]

    # 검지~소지
    for a in m.finger_angles:
        states.append(int(a > cfg.finger_extended_deg))
        margins.append(abs(a - cfg.finger_extended_deg) / cfg.margin_deg)

    return tuple(states), tuple(margins)


def _conf(margins) -> float:
    return float(np.clip(min(margins), 0.0, 1.0))


def finger_states(norm: np.ndarray, cfg: RuleConfig) -> tuple[tuple, float]:
    """정규화 랜드마크 → (손가락 펴짐 패턴, 신뢰도).

    신뢰도 = 가장 애매한 손가락이 임계값에서 얼마나 떨어져 있는지 (0~1).
    한 손가락이라도 임계값 근처면 전체 판별을 믿기 어렵다.
    """
    states, margins = finger_margins(norm, cfg)
    return states, _conf(margins)


def _matches(fingers, pattern) -> bool:
    return all(p is None or p == f for f, p in zip(fingers, pattern))


def classify_normalized(norm: np.ndarray, cfg: RuleConfig | None = None) -> RuleResult:
    """신뢰도는 그 제스처를 정하는 데 쓴 손가락만으로 계산한다.
    (예: V자에서 엄지가 애매해도 V자 판정은 확실하다. OK에서 반쯤 굽힌 검지는 판정에 안 쓴다.)
    """
    cfg = cfg or RuleConfig()
    fingers, margins = finger_margins(norm, cfg)

    # OK: 엄지 펴짐 + 엄지끝·검지끝 맞닿음 + 중지·약지·소지 펴짐 (손가락 패턴보다 먼저 검사).
    # 검지는 반쯤 굽은 상태라 판정에 안 쓴다. 엄지 조건이 없으면 엄지·검지를 둘 다 접은 손(00111)도
    # 두 끝이 가까워 OK가 된다 (11/21 실측 OK의 엄지는 거리 0.97~1.15로 펴짐).
    touch = float(np.linalg.norm(norm[4] - norm[8]))
    if touch < cfg.ok_touch_dist and fingers[0] == 1 and fingers[2:] == (1, 1, 1):
        touch_margin = (cfg.ok_touch_dist - touch) / cfg.thumb_dist_margin
        return RuleResult("ok", _conf((margins[0],) + margins[2:] + (touch_margin,)), fingers)

    for name, pattern in GESTURE_PATTERNS.items():
        if _matches(fingers, pattern):
            used = [m for m, p in zip(margins, pattern) if p is not None]
            return RuleResult(name, _conf(used), fingers)
    return RuleResult(NONE, _conf(margins), fingers)


def classify(landmarks: np.ndarray, handedness: str, aspect: float, cfg: RuleConfig | None = None) -> RuleResult:
    """원본 랜드마크 → 제스처 이름."""
    return classify_normalized(normalize_landmarks(landmarks, handedness, aspect), cfg)
