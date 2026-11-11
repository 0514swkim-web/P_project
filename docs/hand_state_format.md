# `/hand/state` 메시지 형식 (초안 v0.1)

작성: 제스처 담당 · 2026-11-11 · **팀 합의 전 제안안**
근거: 과제계획서 4.1(시스템 구조, 클러치 상태 전이, 예외 처리)

## 1. 위치

```
[손동작 인식부]                                  [로봇 제어부]
카메라 → 손 추적(팀원A) → 필터·제스처(팀원A·제스처 담당) ──/hand/state──▶ 명령 생성(팀원B) → 안전 감시 → 로봇
```

- 인식부와 제어부는 **이 토픽 하나로만** 연결된다 (계획서 4.1).
- 발행 주기: 카메라 프레임마다 (약 30 Hz).
- 명령 생성 노드는 **0.2초간 수신이 없으면 일시정지** (계획서 예외 처리 표).

## 2. 메시지 정의 (제안)

패키지명 제안: `hand_interfaces` → `hand_interfaces/msg/HandState.msg`

```
# ---- 공통 ----
std_msgs/Header header        # stamp: 카메라 프레임 캡처 시각 (지연 측정 기준), frame_id: "camera"
uint32 frame_index            # 프레임 번호 (끊김·누락 확인용)

# ---- 손 인식 ----
bool   hand_detected          # 손이 인식됐는가
string handedness             # "Left" / "Right" (거울 반전 후 기준)
float32 detection_confidence  # MediaPipe 손 인식 신뢰도 [0,1]

# ---- 손 위치 (필터 적용 후) ----
geometry_msgs/Point position  # 손 기준점 = 손목·손바닥 기저부(0,5,9,13,17번) 평균
                              # x, y: 정규화 이미지 좌표 [0,1], 거울 반전 후, 원점 = 화면 왼쪽 위
                              # z: 사용 안 함 (0)

# ---- 제어 모드 (클러치 상태 머신 결과) ----
uint8 MODE_IDLE   = 0         # 대기: 로봇 움직이지 않음
uint8 MODE_PLANAR = 1         # 활성: 손 이동 → 로봇 좌우·전후
uint8 MODE_HEIGHT = 2         # 높이 모드: 손 상하 이동 → 로봇 상하 (평면 고정)
uint8 MODE_PAUSED = 3         # 일시정지: 손 소실·신뢰도 저하 (재개 시 활성화 제스처 필요)
uint8 control_mode

bool hold_position            # true면 명령 생성 노드는 현재 위치 유지
                              #  - 정지 제스처 후보 감지 즉시 (0.5초 확정 전)
                              #  - 그리퍼 전환 중 (파지 순간 흔들림 방지)

# ---- 높이 입력 ----
float32 height_input          # MODE_HEIGHT일 때만 유효. 높이 모드 진입 시점 대비 손의 화면 y 변화량
                              # 위로 올리면 +, 정규화 단위 (화면 높이 = 1.0)

# ---- 그리퍼 ----
uint8 GRIPPER_UNKNOWN = 0
uint8 GRIPPER_OPEN    = 1
uint8 GRIPPER_CLOSED  = 2
uint8 gripper_state           # 안정화(연속 프레임 확인) 후 값. 높이 모드·대기 중에는 직전 값 유지

# ---- 디버그·평가용 ----
string  gesture               # 현재 프레임 판별 결과 (예: "palm", "fist", "clutch", "height", "none")
float32 gesture_confidence    # 판별 신뢰도 [0,1]
```

## 3. 동작 규칙 (명령 생성 노드 입장)

| 수신 내용 | 명령 생성 노드 동작 |
|---|---|
| `control_mode` IDLE / PAUSED | 로봇 정지 (현재 위치 유지) |
| IDLE → PLANAR 전환 순간 | 그 순간의 손 위치·로봇 위치를 **기준점**으로 저장 (클러치 기준 상대 매핑) |
| PLANAR | 기준점 대비 `position` 변화량 × 배율 → 로봇 좌우·전후 |
| HEIGHT | `height_input` × 배율 → 로봇 상하, 평면 고정 |
| `hold_position` = true | 모드와 관계없이 위치 명령 고정 |
| `gripper_state` 변화 | 그리퍼 열기/닫기 |
| 0.2초 미수신 | 일시정지 |

## 4. 합의가 필요한 질문

1. **클러치 상태 머신을 어디서 돌릴지**: 인식부(제스처 노드, 이 초안)에서 판단해 `control_mode`로 보내는 안. 제어부가 직접 판단하려면 원시 제스처만 보내면 됨.
2. **좌표 단위**: 정규화 이미지 좌표 [0,1] (해상도 무관) 제안. 픽셀이 편하면 변경.
3. **`height_input`**: 인식부에서 변화량을 계산해 보낼지, 제어부가 `position.y`로 직접 계산할지.
4. **QoS**: 실시간 센서 데이터라 `BEST_EFFORT`, `KEEP_LAST(1)` 제안.
5. **인식부 내부(팀원A → 제스처 담당)**: 같은 노드 안에서 함수 호출로 넘길지, `/hand/landmarks` 토픽으로 나눌지.
6. **메시지 패키지 담당**: `hand_interfaces` 패키지를 누가 만들고 저장소 어디에 둘지.
