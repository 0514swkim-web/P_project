"""클러치 상태 머신 시나리오 테스트 (30 FPS 가정)."""
from src.gesture.state_machine import ClutchStateMachine, Gripper, Mode

DT = 1 / 30


class Sim:
    def __init__(self):
        self.m = ClutchStateMachine()
        self.t = 0.0
        self.out = None

    def run(self, gesture, seconds, hand_ok=True, y=0.5):
        for _ in range(round(seconds / DT)):
            self.out = self.m.update(self.t, hand_ok, gesture, y)
            self.t += DT
        return self.out


def test_starts_idle_and_ignores_gripper_gestures():
    s = Sim()
    assert s.run("palm", 1.0).control_mode == Mode.IDLE
    assert s.run("fist", 1.0).control_mode == Mode.IDLE


def test_clutch_short_does_not_activate():
    s = Sim()
    assert s.run("call", 0.3).control_mode == Mode.IDLE


def test_clutch_half_second_activates():
    s = Sim()
    assert s.run("call", 0.6).control_mode == Mode.PLANAR


def test_clutch_tolerates_brief_glitch():
    s = Sim()
    s.run("call", 0.3)
    s.run("none", 2 * DT)  # 0.07초 오인식
    assert s.run("call", 0.25).control_mode == Mode.PLANAR


def test_clutch_long_gap_resets_timer():
    s = Sim()
    s.run("call", 0.3)
    s.run("palm", 0.3)  # 진짜로 풀었음
    assert s.run("call", 0.3).control_mode == Mode.IDLE


def test_clutch_toggle_needs_release():
    s = Sim()
    s.run("call", 2.0)  # 계속 들고 있어도 다시 꺼지지 않아야 함
    assert s.out.control_mode == Mode.PLANAR
    assert s.out.hold_position  # 클러치를 들고 있는 동안은 위치 고정
    s.run("palm", 0.5)
    assert s.out.control_mode == Mode.PLANAR and not s.out.hold_position
    s.run("call", 0.6)
    assert s.out.control_mode == Mode.IDLE


def test_stop_candidate_holds_immediately():
    s = Sim()
    s.run("call", 0.6)
    s.run("palm", 0.5)
    out = s.run("call", 2 * DT)  # 0.5초 전이라도
    assert out.control_mode == Mode.PLANAR and out.hold_position


def test_gripper_changes_in_planar_only():
    s = Sim()
    s.run("fist", 0.5)
    assert s.out.gripper_state == Gripper.UNKNOWN  # 대기 중에는 그리퍼 무시
    s.run("call", 0.6)
    s.run("palm", 0.3)
    assert s.out.gripper_state == Gripper.OPEN
    s.run("fist", 0.3)
    assert s.out.gripper_state == Gripper.CLOSED


def test_gripper_transition_holds_position():
    s = Sim()
    s.run("call", 0.6)
    s.run("palm", 0.5)
    out = s.run("fist", DT)  # 바뀌기 시작한 첫 프레임
    assert out.hold_position and out.gripper_state == Gripper.OPEN
    out = s.run("fist", 0.2)  # 확정 후
    assert not out.hold_position and out.gripper_state == Gripper.CLOSED


def test_single_glitch_frame_ignored():
    s = Sim()
    s.run("call", 0.6)
    s.run("palm", 0.5)
    s.run("fist", DT)  # 한 프레임 오인식
    out = s.run("palm", 0.2)
    assert out.gripper_state == Gripper.OPEN


def test_height_mode_and_input():
    s = Sim()
    s.run("call", 0.6)
    s.run("palm", 0.3)
    s.run("peace", 0.2, y=0.6)
    assert s.out.control_mode == Mode.HEIGHT
    out = s.run("peace", 0.1, y=0.5)  # 손을 화면 위로 0.1 올림
    assert abs(out.height_input - 0.1) < 1e-9
    assert out.gripper_state == Gripper.OPEN  # 높이 모드에서 그리퍼 유지
    s.run("palm", 0.2)
    assert s.out.control_mode == Mode.PLANAR


def test_hand_lost_pauses_and_needs_clutch():
    s = Sim()
    s.run("call", 0.6)
    s.run("palm", 0.3)
    assert s.run("palm", 0.5, hand_ok=False).control_mode == Mode.PAUSED  # lost_grace_s(0.3초) 초과
    assert s.run("palm", 1.0).control_mode == Mode.PAUSED  # 손이 돌아와도 자동 재개 안 함
    assert s.run("call", 0.6).control_mode == Mode.PLANAR


def test_brief_hand_loss_holds_but_keeps_mode():
    """0.3초 미만 인식 끊김: 끊긴 동안은 위치 고정, 돌아오면 클러치 없이 그대로 조작 (11/21 실측 반영)."""
    s = Sim()
    s.run("call", 0.6)
    s.run("palm", 0.3)
    out = s.run("palm", 0.2, hand_ok=False)
    assert out.control_mode == Mode.PLANAR and out.hold_position  # 끊긴 동안 로봇은 멈춤
    assert s.run("palm", 0.2).control_mode == Mode.PLANAR
