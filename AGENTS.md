# AGENTS.md — 제스처 판별 모듈 (The-Hand)

## 사용자
- 제스처 담당 (가천대 SW, The-Hand 팀 7조). 손동작 로봇팔 비접촉 제어 캡스톤에서 **제스처 판별·데이터·성능 평가** 담당.
- 한국어로 대화. 이해·학습도 원함 → 개념 설명을 곁들일 것. 요청 시 단계별로 같이 구현(페어).

## 먼저 읽을 것
- `docs/HANDOFF.md` — 이전 대화 전체 정리 (프로젝트 배경, 장비, 결정, 진행 상황, 할 일, 메일 초안)
- `docs/hand_state_format.md`, `docs/gesture_candidates.md`, `docs/dev_environment.md`, `README.md`

## 작업 규칙
- 가상환경: `.venv` (Python 3.11.9). 실행은 `.venv\Scripts\python` 또는 활성화 후 `python`.
- 테스트: `python -m pytest -q` (웹캠 불필요, 52개 통과 상태)
- 웹캠: 팀 웹캠 **ZIO-C960** = `--camera 1` (내장 0번). 최대 30 FPS, USB 2.0 → MJPG 필수, 손은 50cm 이상.
- 패키지 추가 시 `requirements.txt`에 버전 고정. pandas는 2.3.3 유지 (스마트 앱 컨트롤이 3.x 차단).
- mediapipe 1.2.2 = Tasks API (`mp.solutions` 없음). 모델 `assets/hand_landmarker.task`.
- 가짜 데이터(`data/synthetic`, `results/synthetic`) 숫자는 보고서에 쓰지 말 것.
- `data/`, `models/`, `results/`, `.venv/`, `assets/*.task`는 git 제외.

## 현재 상태 (2026-11-11)
- 1·2단계 초안 → 팀 합의 대기 / 3~8단계 코드 완성(실데이터 미검증) / 9단계(ROS 2) 미착수
- WSL2 시뮬레이션 환경 완료(11/11): ROS 2 Jazzy + Gazebo + OMX `omx_f_gazebo.launch.py` 실행·`/joint_states` 확인. 상세·주의점은 `docs/dev_environment.md` W절
- 다음: ZIO 웹캠 확인 → 엄지 등 규칙 임계값 튜닝 → 데이터 수집 → 클러치 후보 비교
- 사용자에게 3~8단계 코드 + 11/11 추가분(베이스라인·참여자 그룹) 설명 예정 (요청받음, 아직 안 함)
- 계획: `docs/requirements_check.pdf`(v2) 기준. 최종 제출 12/13 23:59, 참여자 ID T/S/E
