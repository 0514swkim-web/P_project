# 개발 환경 가이드 (제스처 담당 노트북)

작성: 2026-11-11

## 전체 그림

| 단계 | OS | 용도 | 언제 | 상태 |
|---|---|---|---|---|
| A | Windows | 제스처 판별 개발 (데이터 수집, 판별기, 분류기, 평가) | 지금 | ✅ 완료 |
| B | Ubuntu 24.04 (듀얼부팅) | ROS 2 노드 연동, Gazebo 시뮬레이션, 실물 로봇 | 9단계 전 (실물 이식 13주차 전) | ⬜ |
| C | 공통 | GitHub 팀 저장소 협업 | 코드가 생기면 | ⬜ |
| W | WSL2 Ubuntu 24.04 | 시뮬레이션만 (ROS 2 Jazzy, Gazebo 8.15, RViz2, OMX 워크스페이스) | 지금 | ✅ 11/11 빌드 완료 |

### W. WSL2 시뮬레이션 환경 (11/11)
- 설치: `scripts/setup/wsl_ros2_jazzy.sh` (root) → `scripts/setup/wsl_omx_workspace.sh` (사용자 wook). 워크스페이스 `~/ros2_ws`, open_manipulator jazzy 커밋 `17efe42`
- 실행은 시작 메뉴의 **Ubuntu 24.04 터미널**에서: `ros2 launch open_manipulator_bringup omx_f_gazebo.launch.py`
- 주의 1: PowerShell에서 `wsl ... bash -c`로 실행하면 `~/.bashrc`가 적용되지 않음 → `source /opt/ros/jazzy/setup.bash`를 직접 붙일 것 (안 붙이면 `No module named 'rosidl_generator_py'`)
- 주의 2: colcon 병렬 빌드 기본값이면 WSL 메모리 부족으로 멈춤 → `MAKEFLAGS=-j2 --parallel-workers 2`
- 주의 3: 콘솔 창 클릭 시 "선택" 모드로 출력이 멈춤 → Esc
- 실물 로봇(USB)은 B(듀얼부팅)에서. `.venv`(Windows)와 ROS(WSL)는 별개 환경

---

## A. Windows 환경 ✅

- 위치: `C:\dev\P_project`
- Python 3.11.9 + 가상환경 `.venv`
- 패키지: `requirements.txt` (버전 고정)
- 손 모델: `assets/hand_landmarker.task`
- 확인: `python scripts\check_env.py` (ZIO 웹캠은 `--camera 1`)

새 PC에서 다시 만들 때는 `README.md`의 환경 설정 순서를 따른다.

주의
- 스마트 앱 컨트롤 때문에 pandas는 2.3.3 (3.x 차단됨)
- 스마트 앱 컨트롤 때문에 matplotlib는 3.10.9 (11/21: 3.11.2의 ft2font DLL 차단 → mediapipe import 실패. mediapipe가 내부에서 matplotlib을 불러옴)
- OpenCV는 `opencv-contrib-python`만
- mediapipe 1.x는 Tasks API (`mp.solutions` 없음)

---

## B. Ubuntu 환경 (나중에)

### B-1. 설치 전 확인 (Windows에서)
- [ ] CPU가 인텔/AMD인가 (설정 → 시스템 → 정보)
- [ ] 디스크 여유 **최소 50GB, 권장 100GB 이상**
- [ ] **BitLocker 복구 키** 확인 (Microsoft 계정 → 장치 → BitLocker 키)
- [ ] 중요 파일 백업
- [ ] 8GB 이상 USB 메모리 (설치 USB용)

### B-2. Ubuntu 24.04 듀얼부팅
1. Ubuntu 24.04 LTS ISO 다운로드 → Rufus 등으로 설치 USB 만들기
2. Windows 디스크 관리에서 파티션 축소로 빈 공간 만들기
3. USB로 부팅 → "Install Ubuntu alongside Windows" 또는 빈 공간에 설치
4. 설치 후 부팅 시 Windows / Ubuntu 선택 화면 확인

### B-3. ROS 2 Jazzy + 시뮬레이션
- ROS 2 Jazzy Desktop (공식 문서의 apt 설치 순서대로)
- Gazebo Harmonic + ROS 연동: `ros-jazzy-ros-gz`
- `gz_ros2_control` (멘토님 권장)
- RViz2 (Desktop 설치에 포함)
- ROBOTIS OMX 패키지: 팀원C가 쓰는 버전·브랜치 그대로 맞추기

### B-4. 제스처 모듈 (Ubuntu)
- Ubuntu 24.04 기본 Python은 **3.12** → mediapipe 1.2.2가 설치되는지 **먼저 확인**
- ROS 2(`rclpy`)와 함께 쓰려면 가상환경을 `--system-site-packages`로 만들어 ROS 패키지가 보이게 함
- `requirements.txt` 설치 → `check_env.py`로 ZIO 웹캠 확인 (Ubuntu는 UVC라 드라이버 불필요)

---

## C. 협업

- 팀 저장소: `In-Hyeon/hand-gesture-robot-control`
- 규칙: `main` 직접 push 금지, `feature/기능명` 브랜치 → Pull Request
- 데이터·모델·가상환경은 `.gitignore`로 제외
