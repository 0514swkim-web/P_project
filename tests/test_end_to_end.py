"""가짜 데이터로 수집 CSV → 학습 → 평가 전체 흐름이 끊기지 않는지 확인."""
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic_data import main as make_data  # noqa: E402

from src.gesture.dataio import load_csvs  # noqa: E402
from src.gesture.evaluation import control_metrics  # noqa: E402
from src.gesture.pipeline import GestureRecognizer  # noqa: E402
from src.gesture.training import build_dataset, cross_validate  # noqa: E402


def test_pipeline_runs(tmp_path):
    make_data(out_dir=tmp_path, n_participants=3, seed=1)
    df = load_csvs(sorted(glob.glob(str(tmp_path / "*.csv"))))

    X, y, groups, rule_pred, _ = build_dataset(df)
    assert X.shape[1] == 78 and "none" in set(y)
    assert not any(str(lab).startswith("nonop_") for lab in y)

    results, _ = cross_validate(X, y, groups, rule_pred, n_splits=3)
    assert results["split"].startswith("GroupKFold")
    assert set(results["models"]) == {"rules", "random_forest", "mlp"}

    c = control_metrics(df, lambda aspect: GestureRecognizer(aspect), clutch="call")
    assert c["n_nonop_trials"] == 3 * 5 * 4
    assert c["n_activate_trials"] == 3 * 10  # activate_call만 (rock·ok 시도는 제외)
    assert c["false_activation_rate"][0] <= 0.1
    assert c["activation_failure_rate"][0] <= 0.3
