"""features.py 테스트 — 웹캠 없이 가짜 손 좌표로 확인한다.

실행: python -m pytest -q
"""
import numpy as np
import pytest

from src.gesture.features import (
    FEATURE_DIM,
    feature_vector,
    joint_angles,
    normalize_landmarks,
    palm_center,
)

ASPECT = 4 / 3  # 640x480


def make_hand(curl_fingers=(), center=(0.5, 0.6), size=0.1):
    """화면에 세운 가짜 오른손 (손가락이 위쪽을 향함, y는 아래로 갈수록 커짐).

    curl_fingers에 넣은 손가락(1=검지 … 4=소지)은 PIP에서 90도 꺾는다.
    좌표는 '등방 단위'로 만든 뒤 정규화 이미지 좌표로 되돌린다 (y *= aspect).
    """
    cx, cy = center
    pts = np.zeros((21, 3))
    pts[0] = [0, 0, 0]  # 손목
    # 엄지: 왼쪽 위로 비스듬히 일직선
    for i, k in enumerate(range(1, 5), start=1):
        pts[k] = [-0.35 * i * size, -0.35 * i * size, 0]
    # 검지~소지: 뿌리(MCP)에서 위로 일직선
    for f, base in enumerate((5, 9, 13, 17)):
        x = (-0.3 + 0.2 * f) * size
        pts[base] = [x, -1.0 * size, 0]
        pts[base + 1] = [x, -1.4 * size, 0]
        if f + 1 in curl_fingers:  # PIP에서 화면 안쪽(z)으로 90도 꺾기
            pts[base + 2] = [x, -1.4 * size, -0.3 * size]
            pts[base + 3] = [x, -1.4 * size, -0.6 * size]
        else:
            pts[base + 2] = [x, -1.7 * size, 0]
            pts[base + 3] = [x, -2.0 * size, 0]
    pts[:, 0] += cx
    pts[:, 1] = pts[:, 1] * ASPECT + cy
    return pts


def test_wrist_becomes_origin():
    norm = normalize_landmarks(make_hand(), "Right", ASPECT)
    assert np.allclose(norm[0], 0)


def test_palm_size_is_one():
    norm = normalize_landmarks(make_hand(), "Right", ASPECT)
    assert np.linalg.norm(norm[9, :2]) == pytest.approx(1.0)


def test_position_invariant():
    a = feature_vector(make_hand(center=(0.3, 0.7)), "Right", ASPECT)
    b = feature_vector(make_hand(center=(0.7, 0.5)), "Right", ASPECT)
    assert np.allclose(a, b)


def test_distance_invariant():
    near = feature_vector(make_hand(size=0.15), "Right", ASPECT)
    far = feature_vector(make_hand(size=0.05), "Right", ASPECT)
    assert np.allclose(near, far)


def test_left_hand_mirrored_to_right():
    right = make_hand()
    left = right.copy()
    left[:, 0] = 2 * 0.5 - left[:, 0]  # 화면 중앙 기준 좌우 반전 = 왼손
    a = feature_vector(right, "Right", ASPECT)
    b = feature_vector(left, "Left", ASPECT)
    assert np.allclose(a, b)


def test_straight_finger_angles_near_180():
    angles = joint_angles(normalize_landmarks(make_hand(), "Right", ASPECT))
    index_pip, index_dip = angles[4], angles[5]  # 검지 = 3~5번째 (MCP, PIP, DIP)
    assert index_pip == pytest.approx(180, abs=1)
    assert index_dip == pytest.approx(180, abs=1)


def test_curled_finger_pip_near_90():
    angles = joint_angles(normalize_landmarks(make_hand(curl_fingers=(1,)), "Right", ASPECT))
    assert angles[4] == pytest.approx(90, abs=1)  # 검지 PIP


def test_aspect_matters():
    """단위를 안 맞추면(aspect=1) 각도가 틀어진다 → to_isotropic이 필요한 이유."""
    hand = make_hand()
    hand[8] = hand[7] + np.array([0.05, 0.05 * ASPECT, 0])  # 검지 끝을 실제 45도로 꺾음
    good = joint_angles(normalize_landmarks(hand, "Right", ASPECT))[5]
    bad = joint_angles(normalize_landmarks(hand, "Right", 1.0))[5]
    assert abs(good - bad) > 3


def test_palm_center_ignores_finger_curl():
    open_hand = palm_center(make_hand())
    fist_like = palm_center(make_hand(curl_fingers=(1, 2, 3, 4)))
    assert np.allclose(open_hand, fist_like)


def test_feature_dim():
    assert feature_vector(make_hand(), "Right", ASPECT).shape == (FEATURE_DIM,)


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros((20, 3)), "Right", ASPECT)
