"""가짜 손 좌표 생성기 — 웹캠 없이 판별기·상태 머신·학습 파이프라인을 시험하기 위한 도구.

실제 손과 완전히 같지는 않다. "코드가 의도대로 동작하는지" 확인용이며,
임계값(각도·거리) 튜닝은 반드시 실제 웹캠 데이터로 다시 해야 한다.
"""
from __future__ import annotations

import numpy as np

# 손가락 펴짐 패턴 (엄지, 검지, 중지, 약지, 소지)
PATTERNS = {
    "palm": (1, 1, 1, 1, 1),
    "fist": (0, 0, 0, 0, 0),
    "peace": (0, 1, 1, 0, 0),
    "call": (1, 0, 0, 0, 1),
    "rock": (0, 1, 0, 0, 1),
    "like": (1, 0, 0, 0, 0),
    "one": (0, 1, 0, 0, 0),
    "ok": None,  # 엄지끝·검지끝 맞닿음 + 중지·약지·소지 펴짐 (특수 처리)
    "none": None,  # 표에 없는 아무 손 모양 (무작위)
}
# (1,1,1,0,0)은 11/21부터 V자(엄지 무관)로 판정하므로 none 예시에서 뺌
_NONE_PATTERNS = [(0, 1, 1, 1, 0), (0, 0, 0, 1, 1), (1, 1, 1, 1, 0), (0, 1, 1, 1, 1), (1, 1, 0, 0, 0), (0, 0, 1, 1, 1)]


def _hand_iso(pattern, ok=False):
    """손바닥 크기 1인 오른손, 등방 단위. 손가락은 위(-y)를 향한다."""
    pts = np.zeros((21, 3))
    thumb_ext = pattern[0] if pattern else 0
    if thumb_ext:
        for i, k in enumerate(range(1, 5), start=1):
            pts[k] = [-0.35 * i, -0.35 * i, 0]
    else:  # 손바닥 쪽으로 접힌 엄지
        pts[1] = [-0.30, -0.25, 0]
        pts[2] = [-0.45, -0.55, 0]
        pts[3] = [-0.20, -0.75, 0]
        pts[4] = [0.05, -0.75, 0]

    for f, base in enumerate((5, 9, 13, 17)):
        x = -0.3 + 0.2 * f
        ext = pattern[f + 1] if pattern else (0 if f == 0 else 1)
        pts[base] = [x, -1.0, 0]
        pts[base + 1] = [x, -1.4, 0]
        if ext:
            pts[base + 2] = [x, -1.7, 0]
            pts[base + 3] = [x, -2.0, 0]
        else:  # PIP에서 90도, DIP에서 한 번 더 꺾임
            pts[base + 2] = [x, -1.4, -0.3]
            pts[base + 3] = [x, -1.15, -0.4]

    if ok:  # 11/21 실측 모양: 엄지는 펴서 내밀고, 반쯤 굽힌 검지 끝과 맞닿음 (엄지 거리 약 1.0, 검지 약 145도)
        pts[1:4] = [[-0.35, -0.40, 0], [-0.60, -0.85, 0], [-0.80, -1.40, -0.05]]
        pts[6:9] = [[-0.30, -1.40, 0], [-0.55, -1.70, -0.10], [-0.80, -1.85, -0.15]]
        pts[4] = pts[8] + np.array([-0.05, 0.0, 0.0])
    return pts


def make_hand(
    gesture: str,
    center=(0.5, 0.6),
    size=0.1,
    handedness="Right",
    aspect=4 / 3,
    rotation_deg=0.0,
    noise=0.0,
    rng=None,
):
    """제스처 이름으로 가짜 손 랜드마크 (21, 3)을 만든다. 정규화 이미지 좌표."""
    if gesture not in PATTERNS:
        raise ValueError(f"모르는 제스처: {gesture}")
    rng = rng or np.random.default_rng()
    pattern = PATTERNS[gesture]
    if gesture == "none":
        pattern = _NONE_PATTERNS[rng.integers(len(_NONE_PATTERNS))]
    pts = _hand_iso(pattern, ok=(gesture == "ok"))

    th = np.radians(rotation_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts[:, :2] = pts[:, :2] @ rot.T

    if noise > 0:
        pts += rng.normal(0, noise, pts.shape)

    pts *= size
    if handedness.lower().startswith("left"):
        pts[:, 0] *= -1
    pts[:, 0] += center[0]
    pts[:, 1] = pts[:, 1] * aspect + center[1]
    return pts
