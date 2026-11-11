"""전처리·특징 추출 (구현 4단계).

입력: MediaPipe 손 랜드마크 21개, shape (21, 3)
      x, y = 정규화 이미지 좌표 [0, 1] (x는 화면 너비, y는 화면 높이로 나눈 값)
      z    = 손목 기준 상대 깊이 (x와 비슷한 스케일)
출력: 위치·거리·좌우손과 무관한 "손 모양" 특징 벡터

랜드마크 번호
    0 손목
    엄지 1(CMC) 2(MCP) 3(IP) 4(끝)
    검지 5(MCP) 6(PIP) 7(DIP) 8(끝)
    중지 9 10 11 12 / 약지 13 14 15 16 / 소지 17 18 19 20
"""
from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9
PALM_BASE = (0, 5, 9, 13, 17)  # 손목 + 손가락 뿌리 4개: 손가락을 움직여도 거의 안 움직이는 점들

# 각 손가락의 관절 체인 (손목에서 손끝 방향)
FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
FINGER_NAMES = tuple(FINGER_CHAINS)


def _check_shape(landmarks: np.ndarray) -> np.ndarray:
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"랜드마크 shape은 (21, 3)이어야 합니다. 받은 값: {pts.shape}")
    return pts


def to_isotropic(landmarks: np.ndarray, aspect: float) -> np.ndarray:
    """x와 y의 단위를 같게 맞춘다.

    정규화 좌표는 x를 '너비'로, y를 '높이'로 나눈 값이라 단위가 다르다.
    (640x480 화면에서 x 0.1 = 64px, y 0.1 = 48px)
    이대로 각도·거리를 계산하면 손 모양이 찌그러진 것처럼 계산된다.
    y에 (높이/너비)를 곱해 둘 다 '화면 너비' 단위로 맞춘다. z도 너비 기준이라 그대로 둔다.

    aspect: 화면 너비 / 높이 (640x480 → 4/3, 1920x1080 → 16/9)
    """
    pts = _check_shape(landmarks).copy()
    pts[:, 1] /= aspect
    return pts


def palm_center(landmarks: np.ndarray) -> np.ndarray:
    """로봇 위치 기준점: 손목·손바닥 기저부 평균 (정규화 이미지 좌표 x, y).

    그리퍼 제스처로 손가락을 쥐었다 펴도 이 점은 거의 안 움직인다 (계획서: 그리퍼 전환 중 위치 잠금).
    """
    pts = _check_shape(landmarks)
    return pts[list(PALM_BASE), :2].mean(axis=0)


def normalize_landmarks(landmarks: np.ndarray, handedness: str, aspect: float) -> np.ndarray:
    """손 모양만 남기는 정규화.

    1) x·y 단위 맞추기
    2) 손목을 원점(0,0,0)으로 이동        → 화면 위치와 무관
    3) 손바닥 크기(손목~중지 뿌리 거리)로 나누기 → 카메라 거리와 무관
    4) 왼손이면 x 부호 반전               → 왼손을 오른손 모양으로 통일
    """
    pts = to_isotropic(landmarks, aspect)
    pts -= pts[WRIST]

    palm_size = np.linalg.norm(pts[MIDDLE_MCP, :2])  # 화면 평면 기준 크기 (z는 노이즈가 커서 제외)
    if palm_size < 1e-6:
        raise ValueError("손바닥 크기가 0입니다. 랜드마크가 잘못되었을 수 있습니다.")
    pts /= palm_size

    if handedness.lower().startswith("left"):
        pts[:, 0] *= -1
    return pts


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """b에서의 꺾인 각도 (a-b-c). 일직선이면 180도, 완전히 접히면 0도에 가까움."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def joint_angles(normalized: np.ndarray) -> np.ndarray:
    """손가락 관절 각도 15개 (손가락 5개 × 관절 3개), 단위: 도.

    손가락 체인 (p0, p1, p2, p3, p4)에서 p1, p2, p3 세 관절의 각도.
    예) 검지: 5번(MCP), 6번(PIP), 7번(DIP)
    펴진 손가락은 세 각도가 모두 180도 근처, 접힌 손가락은 작아진다.
    """
    pts = _check_shape(normalized)
    angles = []
    for chain in FINGER_CHAINS.values():
        for i in range(1, 4):
            angles.append(_angle_deg(pts[chain[i - 1]], pts[chain[i]], pts[chain[i + 1]]))
    return np.array(angles)


def feature_vector(landmarks: np.ndarray, handedness: str, aspect: float) -> np.ndarray:
    """분류기 입력: 정규화 좌표 63개 + 관절 각도 15개 = 78차원 (계획서: 정규화 좌표 + 손가락 관절 각도)."""
    norm = normalize_landmarks(landmarks, handedness, aspect)
    angles = joint_angles(norm) / 180.0  # 좌표와 비슷한 크기(0~1)로 맞춤
    return np.concatenate([norm.ravel(), angles])


FEATURE_DIM = NUM_LANDMARKS * 3 + 15
