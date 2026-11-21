"""분류기 학습·비교 (구현 6단계).

실행: python scripts/train_classifier.py --data "data/raw/*.csv"
결과: results/train_report.json, results/confusion_<모델>.png, models/gesture_clf.joblib
"""
import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gesture.dataio import load_csvs  # noqa: E402
from src.gesture.features import FEATURE_DIM  # noqa: E402
from src.gesture.training import build_dataset, cross_validate, for_realtime, inference_ms, make_models  # noqa: E402


def plot_confusion(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "raw" / "*.csv"))
    p.add_argument("--model-out", default=str(ROOT / "models" / "gesture_clf.joblib"))
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--nonop-as-none", action="store_true", help="비조작 동작 프레임도 none 클래스로 학습")
    args = p.parse_args()

    paths = sorted(glob.glob(args.data))
    df = load_csvs(paths)
    X, y, groups, rule_pred, _ = build_dataset(df, nonop_as_none=args.nonop_as_none)
    print(f"파일 {len(paths)}개, 프레임 {len(y)}개, 참여자 {len(set(groups))}명, 클래스 {sorted(set(y))}")

    results, _ = cross_validate(X, y, groups, rule_pred)
    print(f"\n분할: {results['split']}")
    print(f"{'방식':<15}{'정확도':>8}{'macro-F1':>10}")
    for name, r in results["models"].items():
        print(f"{name:<15}{r['accuracy']:>8.3f}{r['macro_f1']:>10.3f}")

    res_dir = Path(args.results)
    res_dir.mkdir(parents=True, exist_ok=True)
    for name, r in results["models"].items():
        plot_confusion(r["confusion"], results["labels"], name, res_dir / f"confusion_{name}.png")

    # 학습 모델을 전체 데이터로 다시 학습 → macro-F1 최고, 같으면 더 빠른 모델을 저장
    fitted, speed = {}, {}
    for name, m in make_models().items():
        fitted[name] = for_realtime(m.fit(X, y))
        speed[name] = inference_ms(fitted[name], X)
        results["models"][name]["inference_ms"] = speed[name]
        print(f"{name:<15} 한 프레임 판별 {speed[name]:.2f} ms (목표 5 ms 미만)")
    best = max(fitted, key=lambda n: (round(results["models"][n]["macro_f1"], 3), -speed[n]))
    model, ms = fitted[best], speed[best]
    results["best_model"] = best
    results["inference_ms"] = ms
    print(f"\n저장 모델: {best}")

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "name": best, "feature_dim": FEATURE_DIM, "classes": list(model.classes_),
         "participants": sorted(set(groups)), "trained_at": datetime.now().isoformat(timespec="seconds")},
        out,
    )
    with open(res_dir / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"결과: {res_dir}  모델: {out}")


if __name__ == "__main__":
    main()
