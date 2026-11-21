"""시간 안정화 + 클러치 상태 머신 (구현 7단계).

프레임마다 판별된 제스처를 받아 /hand/state의 control_mode, hold_position,
height_input, gripper_state를 만든다. 규칙은 과제계획서 4.1을 따른다.

    IDLE(대기) ──클러치 0.5초──▶ PLANAR(평면 이동) ──높이 제스처──▶ HEIGHT(높이)
       ▲                            │  ▲                              │
       └────────클러치 0.5초────────┘  └──────높이 제스처 해제─────────┘
    손 소실·신뢰도 저하: 즉시 위치 고정, 0.3초 넘게 이어지면 PLANAR/HEIGHT → PAUSED (재개하려면 클러치 0.5초)

- 클러치는 토글: 같은 제스처로 시작·정지. 한 번 전환한 뒤에는 손을 풀었다가 다시 해야 다음 전환.
- 활성 중 클러치 후보가 보이면 0.5초 확정 전이라도 즉시 hold_position (정지 의도가 늦게 반영되지 않게).
- 모드·그리퍼 전환은 같은 제스처가 stable_frames 연속일 때만 인정 (순간 오인식 무시).
- 그리퍼가 바뀌는 중(애매한 프레임 포함)에는 hold_position (파지 순간 위치 흔들림 방지).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Mode(IntEnum):
    IDLE = 0
    PLANAR = 1
    HEIGHT = 2
    PAUSED = 3


class Gripper(IntEnum):
    UNKNOWN = 0
    OPEN = 1
    CLOSED = 2


def default_roles():
    # 기능 → 제스처 이름. 클러치 후보를 바꾸려면 여기만 바꾸면 된다 (call / rock / ok).
    return {"gripper_open": "palm", "gripper_close": "fist", "height": "peace", "clutch": "call"}


BASELINE_CLUTCH = "palm"  # 계획서 실험 ②의 비교 기준: 단순 활성화 방식(손바닥 펴기 유지)
_UNUSED = "_unused"  # 어떤 판별 결과와도 같지 않은 제스처 이름 (역할 끄기용)


def roles_for_clutch(clutch: str) -> dict:
    """클러치 후보 하나로 역할표를 만든다.

    clutch가 다른 역할과 같은 제스처면(베이스라인 "palm" = 그리퍼 열림) 그 역할을 끈다.
    이때는 실제 조작용이 아니라 오작동률·활성화 실패율을 같은 데이터로 비교하는 용도다.
    """
    roles = default_roles()
    roles["clutch"] = clutch
    for k in ("gripper_open", "gripper_close", "height"):
        if roles[k] == clutch:
            roles[k] = _UNUSED
    return roles


@dataclass
class ControlConfig:
    roles: dict = field(default_factory=default_roles)
    clutch_hold_s: float = 0.5
    clutch_grace_s: float = 0.15  # 클러치 유지 중 이보다 짧은 끊김(오인식)은 봐준다
    stable_frames: int = 3  # 30 FPS 기준 약 0.1초
    # 손이 이보다 짧게 안 보이면 모드는 유지하고 위치만 고정 (로봇은 즉시 멈춤, 재개에 클러치 불필요).
    # 11/21 실측(야외 10 FPS): 0.07~0.27초 인식 끊김이 잦았고, 매번 PAUSED로 떨어져 클러치를 다시 해야 했다.
    lost_grace_s: float = 0.3


@dataclass
class ControlOutput:
    control_mode: Mode
    hold_position: bool
    height_input: float
    gripper_state: Gripper


class ClutchStateMachine:
    def __init__(self, cfg: ControlConfig | None = None):
        self.cfg = cfg or ControlConfig()
        r = self.cfg.roles
        self._clutch, self._height = r["clutch"], r["height"]
        self._open, self._close = r["gripper_open"], r["gripper_close"]
        self.reset()

    def reset(self):
        self.mode = Mode.IDLE
        self.gripper = Gripper.UNKNOWN
        self._clutch_start = None
        self._clutch_last = None  # 마지막으로 클러치가 보인 시각
        self._clutch_armed = True  # 클러치를 풀었다가 다시 해야 다음 전환 가능
        self._height_y0 = None
        self._run_gesture = None
        self._run_len = 0
        self._lost_since = None  # 손이 안 보이기 시작한 시각

    # ---- 내부 도우미 ----
    def _stable(self, gesture):
        """같은 제스처가 stable_frames 연속이면 그 제스처, 아니면 None (아직 결정 못 함)."""
        if gesture == self._run_gesture:
            self._run_len += 1
        else:
            self._run_gesture, self._run_len = gesture, 1
        return gesture if self._run_len >= self.cfg.stable_frames else None

    def _out(self, hold, height=0.0):
        return ControlOutput(self.mode, hold, height, self.gripper)

    def _gripper_gesture(self):
        return {Gripper.OPEN: self._open, Gripper.CLOSED: self._close}.get(self.gripper)

    def clutch_progress(self, t: float) -> float:
        """클러치 유지가 얼마나 찼는지 0~1 (화면 게이지용). 이미 전환했고 아직 안 풀었으면 0."""
        if self._clutch_start is None or not self._clutch_armed:
            return 0.0
        return min((t - self._clutch_start) / self.cfg.clutch_hold_s, 1.0)

    # ---- 메인 ----
    def update(self, t: float, hand_ok: bool, gesture: str, palm_y: float = 0.0) -> ControlOutput:
        """t: 초 단위 시각, hand_ok: 손이 신뢰할 만하게 보이는가, gesture: 이번 프레임 판별 결과, palm_y: 손 기준점 y."""
        # 1) 손 소실·신뢰도 저하 → 즉시 위치 고정. lost_grace_s 넘게 이어지면 일시정지
        if not hand_ok:
            if self._lost_since is None:
                self._lost_since = t
            if t - self._lost_since < self.cfg.lost_grace_s:
                return self._out(hold=True)  # 잠깐 끊김: 모드·클러치 진행 상태는 그대로
            self._clutch_start = self._clutch_last = None
            self._clutch_armed = True
            self._run_gesture, self._run_len = None, 0
            if self.mode in (Mode.PLANAR, Mode.HEIGHT):
                self.mode = Mode.PAUSED
                self._height_y0 = None
            return self._out(hold=True)
        self._lost_since = None

        stable = self._stable(gesture)

        # 2) 클러치 유지 시간 (짧은 끊김은 유지로 간주)
        is_clutch = gesture == self._clutch
        if is_clutch:
            if self._clutch_start is None:
                self._clutch_start = t
            self._clutch_last = t
        elif self._clutch_last is not None and t - self._clutch_last > self.cfg.clutch_grace_s:
            self._clutch_start = self._clutch_last = None  # 진짜로 풀었음
            self._clutch_armed = True
        in_clutch = self._clutch_start is not None  # 클러치 중 (끊김 허용 구간 포함)
        held = t - self._clutch_start if in_clutch else 0.0
        clutch_fired = is_clutch and self._clutch_armed and held >= self.cfg.clutch_hold_s

        # 3) 대기·일시정지: 클러치 0.5초 → 활성
        if self.mode in (Mode.IDLE, Mode.PAUSED):
            if clutch_fired:
                self.mode = Mode.PLANAR
                self._clutch_armed = False
            return self._out(hold=True)

        # 4) 활성 중 클러치: 즉시 위치 고정, 0.5초 확정되면 대기
        if in_clutch:
            if clutch_fired:
                self.mode = Mode.IDLE
                self._clutch_armed = False
                self._height_y0 = None
            return self._out(hold=True)

        # 5) 평면 ↔ 높이 모드
        if self.mode == Mode.PLANAR and stable == self._height:
            self.mode = Mode.HEIGHT
            self._height_y0 = palm_y
        elif self.mode == Mode.HEIGHT and stable is not None and stable != self._height:
            self.mode = Mode.PLANAR
            self._height_y0 = None

        if self.mode == Mode.HEIGHT:
            return self._out(hold=False, height=self._height_y0 - palm_y)  # 손을 올리면(+y 감소) 양수

        # 6) 평면 모드: 그리퍼
        if stable == self._open:
            self.gripper = Gripper.OPEN
        elif stable == self._close:
            self.gripper = Gripper.CLOSED
        changing = gesture != self._gripper_gesture() and gesture in (self._open, self._close, "none")
        return self._out(hold=changing)
