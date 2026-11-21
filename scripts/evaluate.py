"""평가 (구현 8단계): 제스처 정확도·F1(조건별) + 오작동률 + 활성화 실패율.

실행:
  python scripts/evaluate.py --data "data/raw/*.csv"                       # 규칙 기반
  python scripts/evaluate.py --data "data/raw/*.csv" --model models/gesture_clf.joblib
  python scripts/evaluate.py --data ... --clutch rock                      # 클러치 후보 바꿔 비교
  python scripts/evaluate.py --data ... --clutch palm                      # 베이스라인(단순 활성화)
  python scripts/evaluate.py --data "data/raw/E*.csv" --model ...          # 최종 평가 (평가용 참여자만)
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gesture.dataio import load_csvs, load_participants  # noqa: E402
from src.gesture.evaluation import control_metrics, gesture_metrics, groups_of, non_eval_participants  # noqa: E402
from src.gesture.pipeline import GestureRecognizer  # noqa: E402
from src.gesture.state_machine import BASELINE_CLUTCH, ControlConfig, roles_for_clutch  # noqa: E402
from src.gesture.training import build_dataset  # noqa: E402


def fmt_ci(v):
    m, (lo, hi) = v
    return f"{m * 100:5.1f}%  (95% CI {lo * 100:.1f} ~ {hi * 100:.1f})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "raw" / "*.csv"))
    p.add_argument("--model", default=None, help="학습 모델 경로 (없으면 규칙 기반)")
    p.add_argument("--clutch", default="call",
                   help="클러치 제스처 후보 (call / rock / ok). palm = 베이스라인(단순 활성화: 손바닥 펴기 유지)")
    p.add_argument("--participants", default=str(ROOT / "data" / "participants.csv"),
                   help="참여자 명부 (열: id, group). 없으면 ID 첫 글자 T/S/E로 그룹 판단")
    p.add_argument("--out", default=str(ROOT / "results" / "eval_report.json"))
    args = p.parse_args()

    df = load_csvs(sorted(glob.glob(args.data)))

    # 0) 참여자 그룹 확인: 최종 지표는 평가용(eval) 그룹만으로 (팀원·선정·학습용이 섞이면 점수가 부풀려짐)
    registry = load_participants(args.participants)
    groups = groups_of(df["participant"], registry)
    print("참여자 그룹:", {g: len(ids) for g, ids in groups.items()})
    not_eval = non_eval_participants(df["participant"], registry)
    if not_eval:
        print(f"[참고] 평가용(eval)이 아닌 참여자 {not_eval}가 포함돼 있습니다. 클러치 선정·개발용 평가라면 괜찮지만, "
              "최종 지표로 쓰려면 평가용 참여자 데이터만 넣으세요 (계획서 4.3).")
    bundle = joblib.load(args.model) if args.model else None
    clf = bundle["model"] if bundle else None
    method = bundle["name"] if bundle else "rules"
    if bundle:
        overlap = sorted(set(bundle["participants"]) & set(df["participant"]))
        if overlap:
            print(f"[경고] 학습에 쓴 참여자 {overlap}가 평가 데이터에 있습니다. 점수가 부풀려집니다. "
                  "최종 평가는 학습에 안 쓴 참여자 데이터로 하세요.")

    # 1) 제스처 정확도
    X, y, _, rule_pred, cond = build_dataset(df)
    y_pred = clf.predict(X) if clf is not None else rule_pred
    g = gesture_metrics(y, np.asarray(y_pred), cond)
    print(f"[{method}] 제스처 정확도 {g['accuracy']:.3f} / macro-F1 {g['macro_f1']:.3f}")
    print("  조건별 정확도:", {k: round(v, 3) for k, v in g["by_condition"].items()})

    # 2) 클러치 성능
    roles = roles_for_clutch(args.clutch)
    c = control_metrics(
        df, lambda aspect: GestureRecognizer(aspect, classifier=clf, ctrl_cfg=ControlConfig(roles=roles)), clutch=args.clutch
    )
    baseline = args.clutch == BASELINE_CLUTCH
    print(f"[클러치={args.clutch}{' (베이스라인: 단순 활성화)' if baseline else ''}]")
    if baseline:
        print("  ※ 베이스라인은 손바닥이 그리퍼 열림과 겹쳐 실제 조작용이 아닙니다. 오작동률·활성화 실패율 비교에만 쓰세요.")
    if "false_activation_rate" in c:
        print(f"  오작동률      {fmt_ci(c['false_activation_rate'])}  (목표 5% 이하, trial {c['n_nonop_trials']}회)")
        print("  동작별:", {k: round(v, 3) for k, v in c["false_activation_by_action"].items()})
    if "activation_failure_rate" in c:
        print(f"  활성화 실패율 {fmt_ci(c['activation_failure_rate'])}  (목표 10% 이하, trial {c['n_activate_trials']}회)")
        print(f"  활성화까지 평균 {c['time_to_activate_mean_s']:.2f}초")

    report = {"method": method, "clutch": args.clutch, "baseline": baseline, "participant_groups": groups, "gesture": g,
              **{k: v for k, v in c.items() if k != "trials"}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"결과: {args.out}")


if __name__ == "__main__":
    main()
