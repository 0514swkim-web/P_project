import pytest

from src.gesture.rules import RuleConfig, classify
from src.gesture.synthetic import PATTERNS, make_hand

ASPECT = 4 / 3


@pytest.mark.parametrize("gesture", list(PATTERNS))
def test_each_gesture_recognized(gesture):
    r = classify(make_hand(gesture), "Right", ASPECT)
    assert r.gesture == gesture


@pytest.mark.parametrize("gesture", list(PATTERNS))
def test_left_hand_and_rotation(gesture):
    lm = make_hand(gesture, handedness="Left", rotation_deg=15, center=(0.3, 0.5), size=0.07)
    assert classify(lm, "Left", ASPECT).gesture == gesture


def test_unknown_pattern_is_none():
    # 검지·중지·약지만 편 손(01110)은 표에 없음 → none
    from src.gesture.rules import classify_normalized
    from src.gesture.features import normalize_landmarks

    lm = make_hand("palm")
    norm = normalize_landmarks(lm, "Right", ASPECT)
    # 소지와 엄지를 접힌 모양으로 바꿈
    folded = normalize_landmarks(make_hand("fist"), "Right", ASPECT)
    norm[17:21] = folded[17:21]
    norm[1:5] = folded[1:5]
    assert classify_normalized(norm).gesture == "none"


def test_confidence_range():
    r = classify(make_hand("palm"), "Right", ASPECT)
    assert 0.0 <= r.confidence <= 1.0
    assert r.confidence > 0.5  # 깨끗한 손바닥은 확신이 높아야 함


# 11/21 실측 (T01, 야외 10 FPS, live_demo 녹화 화면에서 읽은 값): (손 모양, 엄지 각도, 엄지 거리, 기대 펴짐)
REAL_THUMBS = [
    ("peace", 152, 0.56, False), ("peace", 151, 0.51, False), ("peace", 150, 0.49, False),
    ("rock", 152, 0.26, False), ("rock", 160, 0.28, False),
    ("one", 152, 0.31, False), ("one", 153, 0.24, False),
    ("fist", 159, 0.34, False),
    ("palm", 173, 1.35, True), ("palm", 172, 1.33, True),
    ("call", 166, 1.45, True), ("call", 166, 1.36, True),
    ("like", 166, 1.18, True), ("like", 167, 1.10, True), ("like", 154, 1.00, True),
    ("ok", 164, 0.97, True), ("ok", 169, 1.15, True),
]


def test_thumb_real_measurements():
    from src.gesture.rules import thumb_state

    cfg = RuleConfig()
    for name, ang, dist, expected in REAL_THUMBS:
        ext, margin = thumb_state(ang, dist, cfg)
        assert ext == expected, (name, ang, dist)
        assert margin > 0.4, (name, ang, dist, margin)  # 신뢰도 컷(0.3)에 안 걸릴 만큼 여유


def test_folded_thumb_confidence_ignores_passing_angle():
    """V자처럼 엄지 각도는 곧은데 거리가 확실히 짧으면, 각도 여유가 신뢰도를 깎지 않는다."""
    from src.gesture.rules import thumb_state

    cfg = RuleConfig()
    _, m_near = thumb_state(cfg.thumb_extended_deg + 1, 0.3, cfg)  # 각도는 겨우 통과, 거리는 확실히 짧음
    assert m_near > 1.0


def test_peace_with_thumb_out():
    """V자는 엄지를 편 채로 해도 V자 (11/21 녹화 D에서 엄지 거리 0.84~1.87로 편 V자가 none 처리됨)."""
    from src.gesture import synthetic as sy

    lm = sy._hand_iso((1, 1, 1, 0, 0))
    lm = lm[:, :] * 0.1 + [0.5, 0.6, 0]  # 이미지 좌표로 (가로세로 비율 1 가정)
    r = classify(lm, "Right", 1.0)
    assert r.gesture == "peace" and r.confidence > 0.3


def test_thumb_and_index_folded_is_not_ok():
    """엄지·검지를 둘 다 접은 손(00111)은 두 끝이 가까워도 OK가 아니다."""
    from src.gesture import synthetic as sy

    lm = sy._hand_iso((0, 0, 1, 1, 1)) * 0.1 + [0.5, 0.6, 0]
    assert classify(lm, "Right", 1.0).gesture != "ok"
