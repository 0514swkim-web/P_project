# Git · GitHub 가이드 (직접 커밋하며 익히기)

작성: 2026-11-11 · 대상: 제스처 담당 · 커밋·푸시는 **직접** 한다. 이 문서는 길잡이만.

---

## 0. 개념 5분 정리

| 용어 | 뜻 | 비유 |
|---|---|---|
| 저장소 (repository) | 파일 + 변경 기록 전체 | 프로젝트 일기장 |
| 커밋 (commit) | "이 시점의 상태"를 메시지와 함께 저장 | 일기 한 편 |
| 스테이징 (`git add`) | 다음 커밋에 넣을 파일 고르기 | 일기에 쓸 사진 고르기 |
| 원격 (remote, GitHub) | 인터넷에 있는 저장소 사본 | 일기장 클라우드 백업 |
| 푸시 (`git push`) | 내 커밋을 GitHub로 올리기 | 백업 업로드 |
| `.gitignore` | Git이 무시할 파일 목록 | "이건 일기에 안 씀" 목록 |

흐름: **파일 수정 → `git add` (고르기) → `git commit` (저장) → `git push` (올리기)**

---

## 1. 처음 한 번만: 준비

### 1-1. 이메일 공개 막기 (먼저!)
커밋에는 작성자 이메일이 박혀서 공개 저장소에서 누구나 볼 수 있다. GitHub가 주는 가짜 주소(noreply)를 쓰자.
1. GitHub → Settings → Emails → **Keep my email addresses private** 체크
2. 거기 표시되는 `숫자+아이디@users.noreply.github.com` 주소를 복사
3. 터미널에서:
```powershell
git config --global user.email "복사한_noreply_주소"
git config --global user.name "0514swkim-web"     # 이미 설정돼 있음. 확인만
git config --global init.defaultBranch main
git config --global core.autocrlf true             # 윈도우 줄바꿈 자동 처리
```

### 1-2. GitHub 로그인 (gh 이미 설치됨)
```powershell
gh auth login      # GitHub.com → HTTPS → 브라우저로 로그인
```

### 1-3. 공개 범위 결정
- **처음엔 private 로 만들기 추천** → 정리가 끝나면 Settings에서 public 전환
- `.gitignore`에서 이미 제외한 것 (11/11): `P-실무/`(과목 원본), `data/`·`models/`·`results/`(참여자 데이터), `docs/HANDOFF.md`·`CLAUDE.md`(연락처 포함 메모), `.vscode/`, `.venv/`
- 팀 저장소(The-Hand 전체)가 따로 생기면, 이 저장소는 **내 담당 모듈 개인 저장소**로 두고 팀 저장소에는 모듈 코드만 PR로 보낸다

---

## 2. 첫 커밋들: 지금까지 한 일을 나눠서 기록

한 번에 `git add .` 하고 "init" 한 줄로 끝내지 말자. **기능 단위로 나눠 커밋**해야 나중에 기록이 이야기가 된다. (지금은 시간 순서가 아니라 "주제별"로 나누면 충분)

```powershell
cd C:\Dev\P_project
git init
git status          # 빨간 파일들 = 아직 안 고른 것. P-실무, data 가 안 보이면 .gitignore 정상
```

추천 순서 (각 단계: `git add ...` → `git status`로 확인 → `git commit -m "..."`):

| # | `git add` 할 것 | 커밋 메시지 예 |
|---|---|---|
| 1 | `.gitignore requirements.txt README.md` | `chore: 프로젝트 초기 설정 (Python 3.11, 패키지 버전 고정)` |
| 2 | `docs/hand_state_format.md docs/gesture_candidates.md` | `docs: /hand/state 메시지 형식과 제스처 후보 정의` |
| 3 | `src/__init__.py src/gesture/__init__.py src/gesture/camera.py src/gesture/pipeline.py src/gesture/features.py scripts/check_env.py scripts/collect_landmarks.py tests/test_features.py` | `feat: MediaPipe 손 랜드마크 수집과 특징 추출` |
| 4 | `src/gesture/rules.py tests/test_rules.py` | `feat: 규칙 기반 제스처 판별기` |
| 5 | `src/gesture/training.py src/gesture/synthetic.py scripts/train_classifier.py scripts/make_synthetic_data.py` | `feat: MLP/RF 분류기 학습과 합성 데이터 생성` |
| 6 | `src/gesture/state_machine.py tests/test_state_machine.py` | `feat: 클러치 상태 머신 (IDLE/PLANAR/HEIGHT/PAUSED)` |
| 7 | `src/gesture/evaluation.py src/gesture/dataio.py scripts/evaluate.py tests/test_end_to_end.py tests/test_baseline_and_groups.py` | `feat: 성능 평가 지표, 베이스라인, 참여자 그룹 분리` |
| 8 | `docs/experiment_protocol.md docs/consent_form.md docs/ai_usage_log.md` | `docs: 실험 프로토콜·동의서 초안, AI 활용 기록` |
| 9 | `docs/*.pdf docs/*.html` | `docs: 구현 계획, 요구사항 점검, 학습 가이드 PDF` |
| 10 | `scripts/setup docs/dev_environment.md docs/git_guide.md` | `build: WSL2 ROS 2 Jazzy + Gazebo OMX 시뮬레이션 환경` |

마지막에 `git status` → "nothing to commit" 이면 다 들어간 것. `git log --oneline` 으로 10줄 확인.

### GitHub에 올리기
```powershell
gh repo create the-hand-gesture --private --source . --push
```
(이름은 자유. 영어 소문자-하이픈 추천)

---

## 3. 평소 작업 루틴 (매번)

```powershell
git status                 # 뭐가 바뀌었나
git diff                   # 구체적으로 어떻게 바뀌었나 (q로 나가기)
git add 파일들              # 고르기
git commit -m "feat: ..."  # 저장
git push                   # 올리기 (하루 끝에 한 번이라도)
```

**좋은 커밋 기준**
- 한 커밋 = 한 가지 이유. "규칙 임계값 튜닝"과 "README 오타"는 따로
- 테스트가 통과하는 상태에서 커밋 (`python -m pytest -q`)
- 메시지는 "무엇을 왜" — `fix: 엄지 판별 임계값 0.6→0.7 (ZIO 웹캠 실측에서 오인식 12%)` 처럼 숫자가 들어가면 최고

**메시지 앞머리 (Conventional Commits)**
| 머리 | 언제 |
|---|---|
| `feat:` | 새 기능 |
| `fix:` | 버그 수정 |
| `docs:` | 문서만 |
| `test:` | 테스트만 |
| `refactor:` | 동작은 그대로, 코드 정리 |
| `exp:` | 실험 결과·튜닝 기록 (이 프로젝트용 자체 규칙) |
| `chore:` / `build:` | 설정·환경 |

VS Code로도 가능: 왼쪽 **소스 제어(Ctrl+Shift+G)** → 파일 옆 `+` = add → 메시지 입력 → ✓ = commit → 동기화 = push. 처음엔 터미널로 익히고 익숙해지면 섞어 쓰기.

---

## 4. 남들이 볼 수 있게: 포트폴리오로 보이는 기록

GitHub 저장소 첫 화면 = README. 면접관·팀원이 **30초 안에** 이해하도록.

### 4-1. README 구성 (프로젝트 끝나갈 때 완성)
1. **한 줄 소개** + 데모 GIF (웹캠 제스처 → Gazebo 로봇팔이 따라 움직이는 화면, 10초)
2. **내 역할** — "7인 캡스톤 중 제스처 판별·데이터·성능 평가 담당"
3. **시스템 구조 그림** — 웹캠 → MediaPipe → 판별기 → 클러치 → `/hand/state` → ROS 2 → OMX
4. **핵심 결과 표** — 실데이터 기준 정확도, 오작동률, 지연 (가짜 데이터 숫자 금지)
5. **기술 결정과 근거** — "규칙 vs MLP 비교 → OO 채택, 이유: …" (ai_usage_log의 '내가 직접 결정한 것' 표가 재료)
6. **실행 방법** — 설치, 테스트, 데모 명령
7. **배운 점 / 어려웠던 점** — 오늘의 WSL 메모리 문제 같은 것도 훌륭한 소재

### 4-2. GitHub 기능 활용
| 기능 | 어떻게 | 왜 보기 좋은가 |
|---|---|---|
| **Issues** | 할 일 하나 = 이슈 하나 ("엄지 임계값 튜닝"). 커밋 메시지에 `#3` 쓰면 자동 연결, `fixes #3`이면 자동 닫힘 | 문제 → 해결 과정이 남음 |
| **Projects (보드)** | Todo / In progress / Done 칸반 | 진행 관리 능력이 보임 |
| **Milestones** | "중간발표", "실물 이식", "최종 12/13" | 일정 관리 |
| **Tags / Releases** | 발표 때마다 `git tag v0.1-midterm` → `git push --tags`, Releases에 PDF 첨부 | 단계별 결과물이 한눈에 |
| **About / Topics** | 저장소 오른쪽 톱니 → 설명, `ros2` `mediapipe` `gesture-recognition` `hri` | 검색·첫인상 |

### 4-3. 기록 습관 (주 1회 10분)
- `docs/` 에 `devlog.md` 하나: 날짜 · 한 일 · 막힌 것 · 결정 → 커밋 `docs: 11/11 개발 일지`
- 실험 결과는 그림(혼동행렬, 그래프)을 `docs/figures/` 에 넣고 README에서 링크 (원본 데이터는 올리지 않음)

---

## 5. 자주 하는 실수와 되돌리기

| 상황 | 해결 |
|---|---|
| add 잘못함 (커밋 전) | `git restore --staged 파일` |
| 방금 커밋 메시지 오타 (push 전) | `git commit --amend -m "새 메시지"` |
| 올리면 안 되는 파일을 커밋함 (push 전) | `.gitignore`에 추가 → `git rm --cached 파일` → 커밋 |
| **이미 push 했는데 개인정보가 들어감** | 저장소를 private로 바꾸고 바로 물어보기 (기록에서 지우는 건 별도 작업) |
| 뭔가 꼬임 | 아무것도 지우지 말고 `git status` 화면을 캡처해서 물어보기 |

push 전에는 거의 모든 게 되돌릴 수 있다. **push 전 `git status` 한 번** 습관만 들이면 된다.
