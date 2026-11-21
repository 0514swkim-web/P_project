"""베이스라인(단순 활성화) 모드와 참여자 그룹 확인 테스트."""
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic_data import main as make_data  # noqa: E402

from src.gesture.dataio import load_csvs, load_participants, participant_group  # noqa: E402
from src.gesture.evaluation import control_metrics, groups_of, non_eval_participants  # noqa: E402
from src.gesture.pipeline import GestureRecognizer  # noqa: E402
from src.gesture.state_machine import (  # noqa: E402
    BASELINE_CLUTCH,
    ClutchStateMachine,
    ControlConfig,
    Gripper,
    Mode,
    roles_for_clutch,
)

DT = 1 / 30


def run(m, gesture, seconds, t0=0.0):
    t, out = t0, None
    for _ in range(round(seconds / DT)):
        out = m.update(t, True, gesture, 0.5)
        t += DT
    return out, t


# ---- 베이스라인 ----
def test_roles_for_clutch_normal_keeps_gripper():
    r = roles_for_clutch("rock")
    assert r["clutch"] == "rock" and r["gripper_open"] == "palm"


def test_roles_for_baseline_disables_overlapping_role():
    r = roles_for_clutch(BASELINE_CLUTCH)
    assert r["clutch"] == "palm"
    assert r["gripper_open"] != "palm"  # 손바닥이 그리퍼 열림으로도 쓰이면 안 됨
    assert r["gripper_close"] == "fist" and r["height"] == "peace"


def test_baseline_palm_hold_activates():
    m = ClutchStateMachine(ControlConfig(roles=roles_for_clutch("palm")))
    out, _ = run(m, "palm", 0.6)
    assert out.control_mode == Mode.PLANAR


def test_baseline_palm_never_sets_gripper_open():
    m = ClutchStateMachine(ControlConfig(roles=roles_for_clutch("palm")))
    out, t = run(m, "palm", 0.6)
    out, t = run(m, "none", 0.3, t)  # 손바닥을 풀고
    out, t = run(m, "fist", 0.3, t)
    assert out.gripper_state == Gripper.CLOSED
    assert out.control_mode == Mode.PLANAR


def test_baseline_counts_palm_activation_trials(tmp_path):
    make_data(out_dir=tmp_path, n_participants=2, seed=2)
    df = load_csvs(sorted(glob.glob(str(tmp_path / "*.csv"))))
    roles = roles_for_clutch("palm")
    c = control_metrics(df, lambda a: GestureRecognizer(a, ctrl_cfg=ControlConfig(roles=roles)), clutch="palm")
    assert c["n_activate_trials"] == 2 * 10  # activate_palm만
    # 손 흔들기(손바닥 유지)는 베이스라인에서 오작동을 일으켜야 한다 → 설계 클러치와 비교하는 이유
    assert c["false_activation_by_action"]["nonop_wave"] > 0.5


# ---- 참여자 그룹 ----
def test_group_from_prefix():
    assert participant_group("T01") == "team"
    assert participant_group("S03") == "select"
    assert participant_group("E08") == "eval"
    assert participant_group("P01") == "unknown"


def test_registry_overrides_prefix(tmp_path):
    p = tmp_path / "participants.csv"
    p.write_text("id,group\nP01,eval\nE02,select\n", encoding="utf-8")
    reg = load_participants(p)
    assert participant_group("P01", reg) == "eval"
    assert participant_group("E02", reg) == "select"
    assert load_participants(tmp_path / "missing.csv") == {}


def test_non_eval_participants_flags_mixed_data():
    ids = ["E01", "E02", "S01", "T01", "E01"]
    assert groups_of(ids) == {"eval": ["E01", "E02"], "select": ["S01"], "team": ["T01"]}
    assert sorted(non_eval_participants(ids)) == ["S01", "T01"]
    assert non_eval_participants(["E01", "E02"]) == []
