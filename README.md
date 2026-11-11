# 제스처 판별 모듈 (The-Hand)

웹캠 손 랜드마크(MediaPipe 21점)로 그리퍼 열림/닫힘, 클러치(시작·정지), 높이 모드, 비조작 동작을 판별하는 모듈.
담당: 제스처 담당

## 폴더 구조

| 폴더 | 내용 |
|---|---|
| `src/gesture/` | 모듈 코드 (전처리·특징·판별기·상태 머신) |
| `scripts/` | 실행 스크립트 (데이터 수집, 학습, 평가) |
| `data/raw/` | 수집한 랜드마크 CSV (영상은 저장하지 않음) |
| `data/processed/` | 전처리·특징 추출 결과 |
| `models/` | 학습된 분류기 |
| `results/` | 평가 결과 (지표, 그래프) |
| `notebooks/` | 실험용 노트북 |
| `tests/` | 테스트 |
| `docs/` | 문서 (제스처 정의, /hand/state 형식 등) |

## 환경 설정 (Windows)

```powershell
cd C:\dev\P_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# MediaPipe 손 모델 (mediapipe 1.x는 Tasks API라 모델 파일이 필요)
curl.exe -L -o assets\hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# 환경 확인 (웹캠 창이 뜨고 손에 21개 점이 그려지면 성공, q로 종료)
python scripts\check_env.py
```

## 작업 흐름

```powershell
python -m pytest -q                                   # 테스트 (웹캠 불필요)

# 1) 수집 (ZIO 웹캠 = --camera 1, 손은 카메라에서 50cm 이상)
python scripts\collect_landmarks.py --participant T01 --condition bright_front --camera 1
python scripts\collect_landmarks.py --participant T01 --camera 1 --labels nonop_drink,nonop_head,nonop_wave,nonop_phone,nonop_point,activate_call,activate_rock,activate_ok,activate_palm

# 2) 학습·비교 (규칙 vs 랜덤 포레스트 vs MLP, 참여자 단위 교차검증)
python scripts\train_classifier.py --data "data\raw\*.csv"

# 3) 평가 (정확도·F1, 오작동률, 활성화 실패율) — 최종 평가는 학습에 안 쓴 참여자로
python scripts\evaluate.py --data "data\raw\*.csv" --clutch call
python scripts\evaluate.py --data "data\raw\*.csv" --model models\gesture_clf.joblib

# 웹캠 없이 전체 흐름 시험 (가짜 데이터, 숫자는 의미 없음)
python scripts\make_synthetic_data.py
python scripts\train_classifier.py --data "data\synthetic\*.csv" --model-out models\synthetic_clf.joblib --results results\synthetic
```

| 모듈 | 단계 | 역할 |
|---|---|---|
| `src/gesture/features.py` | 4 | 정규화(손목 원점·손바닥 크기·왼손 반전), 관절 각도, 특징 78차원 |
| `src/gesture/rules.py` | 5 | 손가락 펴짐 패턴 → 제스처 (규칙 기반 기준선) |
| `src/gesture/training.py` | 6 | 데이터셋 구성, GroupKFold 비교, 판별 시간 |
| `src/gesture/state_machine.py` | 7 | 연속 프레임 안정화, 클러치 0.5초 토글, 높이 모드, 그리퍼 |
| `src/gesture/pipeline.py` | 4~7 | 프레임 하나 → `/hand/state` 값 |
| `src/gesture/evaluation.py` | 8 | 오작동률, 활성화 실패율(참여자 단위 95% CI), 조건별 정확도 |
| `src/gesture/camera.py` | 3 | 웹캠(MJPG) 열기, MediaPipe 손 인식 |
| `src/gesture/dataio.py` | 3 | 수집 CSV 형식 |
| `src/gesture/synthetic.py` | — | 테스트용 가짜 손 |

주의
- Windows 스마트 앱 컨트롤이 켜져 있으면 pandas 3.x가 차단됨 → 2.3.3 사용.
- OpenCV는 `opencv-contrib-python`만 설치 (`opencv-python`과 같이 깔면 충돌).
- mediapipe 1.x에는 예전 `mp.solutions.hands`가 없음 → `mediapipe.tasks.python.vision.HandLandmarker` 사용.
