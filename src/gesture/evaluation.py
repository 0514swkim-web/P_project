"""평가 지표 (구현 8단계).

- 오작동률: 비조작 동작 1회(trial) 중 활성화가 한 번이라도 일어나면 오작동 1건
- 활성화 실패율: 의도적 활성화 시도(trial)에서 3초 안에 활성화되지 않은 비율 + 활성화까지 걸린 시간
- 참여자별 비율을 단위로 평균과 t 분포 95% 신뢰구간 (같은 사람의 시행은 서로 독립이 아니므로)
- 제스처 정확도·F1: 조건(조명·각도)별로도 계산
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

from .dataio import ACTIVATE_LABEL, EVAL_GROUP, NONOP_PREFIX, participant_group, row_landmarks
from .state_machine import Mode

ACTIVE = (Mode.PLANAR, Mode.HEIGHT)


def mean_ci(values, confidence=0.95, bounds=(0.0, 1.0)):
    """평균과 t 분포 신뢰구간. 값이 1개면 구간 없음. 비율이라 구간을 [0, 1]로 자른다."""
    v = np.asarray(values, dtype=float)
    m = float(v.mean()) if len(v) else float("nan")
    if len(v) < 2:
        return m, (float("nan"), float("nan"))
    half = stats.t.ppf((1 + confidence) / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    lo, hi = m - half, m + half
    if bounds:
        lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    return m, (lo, hi)


def groups_of(participants, registry: dict | None = None) -> dict:
    """{그룹: [참여자 ID…]}. 최종 지표는 평가용(eval) 그룹만으로 내야 한다 (계획서 4.3)."""
    out: dict = {}
    for pid in sorted(set(map(str, participants))):
        out.setdefault(participant_group(pid, registry), []).append(pid)
    return out


def non_eval_participants(participants, registry: dict | None = None) -> list:
    """평가용이 아닌 참여자 (팀원·선정·학습용·미분류). 이들이 섞이면 최종 지표로 쓸 수 없다."""
    return [p for g, ids in groups_of(participants, registry).items() if g != EVAL_GROUP for p in ids]


def run_trial(recognizer, trial: pd.DataFrame):
    """한 trial의 프레임을 순서대로 판별기에 넣고 HandState 목록을 돌려준다."""
    recognizer.reset()
    states = []
    for _, r in trial.sort_values("frame_idx").iterrows():
        states.append(recognizer.process(float(r["t"]), row_landmarks(r), str(r["handedness"]), float(r["score"])))
    return states


def control_metrics(df: pd.DataFrame, make_recognizer, clutch: str = "call", activation_window_s: float = 3.0):
    """make_recognizer(aspect) → GestureRecognizer. 비조작·활성화 trial로 오작동률·활성화 실패율 계산.

    활성화 trial 라벨: "activate"(기본 클러치) 또는 "activate_<제스처>"(후보별로 따로 녹화한 시도).
    clutch와 맞는 활성화 trial만 실패율 계산에 쓴다.
    """
    records = []
    keys = ["participant", "session", "trial_id"]
    for (pid, _, _), trial in df.groupby(keys):
        label = str(trial["label"].iloc[0])
        is_nonop = label.startswith(NONOP_PREFIX)
        is_activate = label == ACTIVATE_LABEL or label == f"{ACTIVATE_LABEL}_{clutch}"
        if not (is_nonop or is_activate):
            continue
        aspect = float(trial["width"].iloc[0]) / float(trial["height"].iloc[0])
        states = run_trial(make_recognizer(aspect), trial)
        active_t = next((s.stamp for s in states if s.control_mode in ACTIVE), None)
        records.append(
            {
                "participant": pid,
                "label": label,
                "kind": "nonop" if is_nonop else "activate",
                "activated": active_t is not None,
                "time_to_activate": active_t,
                "failed": (active_t is None or active_t > activation_window_s) if not is_nonop else None,
            }
        )
    rec = pd.DataFrame(records)
    out = {"trials": rec}
    if rec.empty:
        return out

    nonop = rec[rec["kind"] == "nonop"]
    if not nonop.empty:
        per_p = nonop.groupby("participant")["activated"].mean()
        out["false_activation_rate"] = mean_ci(per_p.values)
        out["false_activation_by_action"] = nonop.groupby("label")["activated"].mean().to_dict()
        out["n_nonop_trials"] = len(nonop)

    act = rec[rec["kind"] == "activate"]
    if not act.empty:
        per_p = act.groupby("participant")["failed"].mean()
        out["activation_failure_rate"] = mean_ci(per_p.values.astype(float))
        ok_times = act.loc[~act["failed"].astype(bool), "time_to_activate"]
        out["time_to_activate_mean_s"] = float(ok_times.mean()) if len(ok_times) else float("nan")
        out["n_activate_trials"] = len(act)
    return out


def gesture_metrics(y_true, y_pred, conditions=None):
    labels = sorted(set(y_true))
    res = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": dict(zip(labels, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).round(3).tolist())),
    }
    if conditions is not None:
        res["by_condition"] = {}
        for c in sorted(set(conditions)):
            m = conditions == c
            res["by_condition"][c or "(없음)"] = float(accuracy_score(y_true[m], y_pred[m]))
    return res
