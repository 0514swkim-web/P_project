"""경량 분류기 학습·비교 (구현 6단계).

- 입력: 수집 CSV → 특징 78차원 (정규화 좌표 63 + 관절 각도 15)
- "none" 클래스 = 명령이 아닌 손 모양을 label "none"으로 따로 녹화한 프레임
- 비조작 동작(nonop_*) 프레임은 기본적으로 학습에서 뺀다.
  손 흔들기처럼 모양 자체가 손바닥인 동작이 있어, 프레임마다 "none"을 붙이면
  같은 모양에 다른 정답을 주게 된다. 비조작 동작은 오작동률 평가(evaluation.py)에 쓴다.
  계획서대로 "none"에 포함해 보려면 nonop_as_none=True.
- 참여자 단위 교차검증(GroupKFold): 같은 사람이 학습과 평가에 동시에 들어가지 않게 한다
- 규칙 기반(학습 없음) vs 랜덤 포레스트 vs MLP 를 같은 분할에서 비교
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataio import ACTIVATE_LABEL, NONOP_PREFIX, row_landmarks
from .features import FEATURE_DIM, feature_vector, normalize_landmarks
from .rules import NONE, classify_normalized


def to_class_label(label: str) -> str:
    return NONE if label.startswith(NONOP_PREFIX) else label


def build_dataset(df: pd.DataFrame, nonop_as_none: bool = False):
    """손이 인식된 프레임만 → X (n, 78), y, groups(참여자), rule_pred(규칙 판별 결과), conditions."""
    keep = (df["hand_detected"] == 1) & ~df["label"].astype(str).str.startswith(ACTIVATE_LABEL)
    if not nonop_as_none:
        keep &= ~df["label"].astype(str).str.startswith(NONOP_PREFIX)
    rows = df[keep]
    X = np.empty((len(rows), FEATURE_DIM))
    rule_pred = []
    for i, (_, r) in enumerate(rows.iterrows()):
        lm = row_landmarks(r)
        aspect = r["width"] / r["height"]
        X[i] = feature_vector(lm, r["handedness"], aspect)
        rule_pred.append(classify_normalized(normalize_landmarks(lm, r["handedness"], aspect)).gesture)
    y = rows["label"].map(to_class_label).to_numpy()
    return X, y, rows["participant"].to_numpy(), np.array(rule_pred), rows["condition"].to_numpy()


def make_models():
    return {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
        "mlp": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=0)
        ),
    }


def _scores(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def cross_validate(X, y, groups, rule_pred, n_splits=5):
    """모델별 교차검증 점수와 혼동행렬. 참여자가 2명 미만이면 경고 후 프레임 단위 분할."""
    labels = sorted(set(y))
    n_groups = len(set(groups))
    if n_groups >= 2:
        splitter = GroupKFold(n_splits=min(n_splits, n_groups))
        splits = list(splitter.split(X, y, groups))
        split_note = f"GroupKFold (참여자 {n_groups}명, {len(splits)}겹)"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        splits = list(splitter.split(X, y))
        split_note = "⚠ 참여자 1명 — 프레임 단위 분할이라 점수가 과대평가됨"

    results = {"split": split_note, "labels": labels, "models": {}}
    preds = {"rules": np.empty(len(y), dtype=object)}
    for name in make_models():
        preds[name] = np.empty(len(y), dtype=object)

    fold_scores = {k: [] for k in preds}
    for train_idx, test_idx in splits:
        fold_scores["rules"].append(_scores(y[test_idx], rule_pred[test_idx], labels))
        preds["rules"][test_idx] = rule_pred[test_idx]
        for name, model in make_models().items():
            model.fit(X[train_idx], y[train_idx])
            p = model.predict(X[test_idx])
            preds[name][test_idx] = p
            fold_scores[name].append(_scores(y[test_idx], p, labels))

    for name, p in preds.items():
        overall = _scores(y, p, labels)
        overall["fold_macro_f1"] = [s["macro_f1"] for s in fold_scores[name]]
        overall["confusion"] = confusion_matrix(y, p.astype(str), labels=labels).tolist()
        results["models"][name] = overall
    return results, preds


def for_realtime(model):
    """실시간용: 한 프레임씩 판별할 때는 병렬 처리 준비 비용이 더 커서 코어 1개만 쓰게 한다."""
    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=1)
    return model


def inference_ms(model, X, n=200):
    """한 프레임 판별 시간 (계획서 목표: 5ms 미만)."""
    idx = np.random.default_rng(0).integers(0, len(X), size=min(n, len(X)))
    start = time.perf_counter()
    for i in idx:
        model.predict_proba(X[i : i + 1])
    return (time.perf_counter() - start) * 1000 / len(idx)
