"""랜드마크 CSV 형식 (영상은 저장하지 않는다).

한 줄 = 한 프레임. 녹화 버튼을 누른 한 구간 = 한 trial.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

META_COLUMNS = [
    "participant",  # 참여자 ID (P01 …) — 학습/평가 분할 단위
    "session",  # 세션 이름
    "condition",  # 촬영 조건 (예: bright_front, dim_side)
    "trial_id",  # 녹화 구간 번호
    "label",  # 제스처(palm, fist, …) 또는 비조작 동작(nonop_drink …) 또는 activate
    "frame_idx",
    "t",  # trial 시작 기준 초
    "width",
    "height",
    "hand_detected",
    "handedness",
    "score",
]
LANDMARK_COLUMNS = [f"{a}{i}" for i in range(21) for a in ("x", "y", "z")]
COLUMNS = META_COLUMNS + LANDMARK_COLUMNS

NONOP_PREFIX = "nonop_"
ACTIVATE_LABEL = "activate"

# 참여자 그룹 (과제계획서 4.3): 팀원은 학습에만, 선정·학습용은 클러치 선정·학습, 평가용은 최종 지표에만.
# ID 첫 글자로 그룹을 정한다: T01(팀원), S01(선정·학습용), E01(평가용). 명부 파일이 있으면 그쪽이 우선.
GROUP_BY_PREFIX = {"T": "team", "S": "select", "E": "eval"}
EVAL_GROUP = "eval"


class CsvRecorder:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(COLUMNS)

    def write(self, meta: dict, landmarks):
        row = [meta.get(c, "") for c in META_COLUMNS]
        row += list(np.asarray(landmarks).ravel()) if landmarks is not None else [""] * len(LANDMARK_COLUMNS)
        self._w.writerow(row)

    def flush(self):
        self._f.flush()

    def close(self):
        self._f.close()


def load_csvs(paths) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        raise FileNotFoundError("CSV 파일이 없습니다.")
    df = pd.concat(frames, ignore_index=True)
    df["participant"] = df["participant"].astype(str)
    df["condition"] = df["condition"].fillna("").astype(str)
    return df


def load_participants(path) -> dict:
    """참여자 명부 CSV(열: id, group) → {id: group}. 파일이 없으면 빈 dict.

    명부에는 이름 같은 개인정보를 넣지 않는다 (이름-ID 대응표는 동의서와 함께 오프라인 보관).
    """
    path = Path(path)
    if not path.exists():
        return {}
    reg = pd.read_csv(path, dtype=str)
    return dict(zip(reg["id"].str.strip(), reg["group"].str.strip()))


def participant_group(pid: str, registry: dict | None = None) -> str:
    """명부 → ID 첫 글자 순서로 그룹을 정한다. 모르면 "unknown"."""
    pid = str(pid)
    if registry and pid in registry:
        return registry[pid]
    return GROUP_BY_PREFIX.get(pid[:1].upper(), "unknown")


def row_landmarks(row) -> np.ndarray | None:
    if not int(row["hand_detected"]):
        return None
    return np.array([row[c] for c in LANDMARK_COLUMNS], dtype=float).reshape(21, 3)
