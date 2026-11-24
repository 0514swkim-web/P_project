"""실시간 데모 처리부(LiveSession·오버레이)를 가짜 검출기로 확인 (웹캠 불필요)."""
import numpy as np

from src.gesture.features import normalize_landmarks
from src.gesture.live import LiveSession, draw_overlay, free_output_paths, state_row
from src.gesture.pipeline import GestureRecognizer
from src.gesture.rules import RuleConfig, finger_metrics, finger_states
from src.gesture.state_machine import Mode
from src.gesture.synthetic import make_hand

ASPECT = 16 / 9
FPS = 30


class FakeDetector:
    """detect()가 불릴 때마다 정해 둔 제스처 손을 돌려준다. None이면 손 없음."""

    def __init__(self):
        self.gesture = None

    def detect(self, frame, t):
        if self.gesture is None:
            return None, "", 0.0
        return make_hand(self.gesture, aspect=ASPECT, rng=np.random.default_rng(0)), "Right", 0.95


def make_session(clutch="call"):
    det = FakeDetector()
    # 가짜 손은 신뢰도가 낮게 나오는 경우가 있어 신뢰도 컷은 끈다 (여기선 흐름만 확인)
    rec = GestureRecognizer(aspect=ASPECT, min_gesture_conf=0.0)
    return det, LiveSession(det, rec, clutch)


def run(session, det, gesture, seconds, t0):
    det.gesture = gesture
    frame = np.zeros((720, 1280, 3), np.uint8)
    r, t = None, t0
    for i in range(int(seconds * FPS)):
        t = t0 + i / FPS
        r = session.step(frame, t)
    return r, t + 1 / FPS


def test_finger_metrics_match_states():
    norm = normalize_landmarks(make_hand("palm", aspect=ASPECT), "Right", ASPECT)
    m = finger_metrics(norm)
    cfg = RuleConfig()
    fingers, _ = finger_states(norm, cfg)
    assert len(m.finger_angles) == 4
    assert fingers[1:] == tuple(int(a > cfg.finger_extended_deg) for a in m.finger_angles)


def test_clutch_activates_and_pauses():
    det, s = make_session("call")
    r, t = run(s, det, "call", 0.3, 0.0)
    assert r.state.control_mode == Mode.IDLE
    assert 0.4 < r.clutch_progress < 1.0  # 0.3초 / 0.5초 정도 찼음

    r, t = run(s, det, "call", 0.4, t)
    assert r.state.control_mode == Mode.PLANAR
    assert r.clutch_progress == 0.0  # 전환 후엔 손을 풀기 전까지 게이지 없음

    r, t = run(s, det, None, 0.5, t)  # lost_grace_s(0.3초)보다 길게
    assert r.state.control_mode == Mode.PAUSED and r.state.hold_position
    assert r.fingers is None and r.metrics is None


def test_switch_clutch():
    det, s = make_session("call")
    assert s.next_clutch() == "rock"
    r, t = run(s, det, "call", 1.0, 0.0)
    assert r.state.control_mode == Mode.IDLE  # call은 더 이상 클러치가 아님
    r, _ = run(s, det, "rock", 0.7, t)
    assert r.state.control_mode == Mode.PLANAR


def test_state_row_and_overlay():
    det, s = make_session()
    for g in ("palm", None):
        r, _ = run(s, det, g, 0.1, 0.0)
        row = state_row(r, s.clutch)
        assert row["control_mode"] in Mode.__members__ and row["clutch"] == "call"
        assert (row["fingers"] == "") == (g is None)  # 손 있을 때만 원시 값 기록
        frame = np.zeros((720, 1280, 3), np.uint8)
        draw_overlay(frame, r, RuleConfig(), s.clutch, fps=30.0, detail=True)
        assert frame.any()  # 뭔가 그려졌음


def test_free_output_paths_never_overwrite(tmp_path):
    mp4, csv = tmp_path / "T01_A.mp4", tmp_path / "T01_A.csv"
    # 없으면 그대로
    assert free_output_paths(str(mp4), str(csv)) == [str(mp4), str(csv)]
    # 하나라도 있으면 둘 다 같은 번호
    csv.write_text("x")
    assert free_output_paths(str(mp4), str(csv)) == [str(tmp_path / "T01_A_2.mp4"), str(tmp_path / "T01_A_2.csv")]
    (tmp_path / "T01_A_2.mp4").write_text("x")
    assert free_output_paths(str(mp4), str(csv))[0] == str(tmp_path / "T01_A_3.mp4")
    # 지정 안 한 쪽은 None 유지
    assert free_output_paths(None, str(csv)) == [None, str(tmp_path / "T01_A_2.csv")]
