"""랜드마크 수집·라벨링 (구현 3단계). 영상은 저장하지 않고 손 좌표만 CSV로 저장한다.

사용 예)
  # 제스처 데이터
  python scripts/collect_landmarks.py --participant T01 --condition bright_front --camera 1
  # 비조작 동작 + 클러치 후보별·베이스라인(palm) 의도적 활성화 시도 (오작동률·활성화 실패율 평가용)
  # 참여자 ID: T01~(팀원) / S01~(선정·학습용) / E01~(평가용) — 첫 글자로 그룹이 정해짐
  python scripts/collect_landmarks.py --participant T01 --labels nonop_drink,nonop_head,nonop_wave,nonop_phone,nonop_point,activate_call,activate_rock,activate_ok,activate_palm

조작
  1~9    라벨 선택 (--labels 순서)
  SPACE  녹화 시작/정지 (한 번 녹화 = 한 trial)
  q/ESC  종료
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gesture.camera import HandDetector, draw_hand, open_camera  # noqa: E402
from src.gesture.dataio import CsvRecorder  # noqa: E402
from src.gesture.rules import classify  # noqa: E402

DEFAULT_LABELS = "palm,fist,peace,call,rock,ok,like,one,none"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--participant", required=True, help="참여자 ID (예: P01)")
    p.add_argument("--session", default="s1")
    p.add_argument("--condition", default="", help="촬영 조건 (예: bright_front)")
    p.add_argument("--labels", default=DEFAULT_LABELS, help="쉼표로 구분, 1~9번 키에 순서대로 배정")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", default=str(ROOT / "data" / "raw"))
    args = p.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()][:9]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / f"{args.participant}_{args.session}_{stamp}.csv"

    cap, w, h, fps = open_camera(args.camera, args.width, args.height)
    print(f"웹캠 {args.camera}: {w}x{h} @ {fps:.0f}fps → 저장: {out}")
    aspect = w / h

    rec = CsvRecorder(out)
    label_i, recording, trial_id, frame_idx = 0, False, 0, 0
    counts = {lab: 0 for lab in labels}
    trial_start = 0.0
    t0 = time.perf_counter()

    with HandDetector() as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임을 읽지 못했습니다.")
                break
            now = time.perf_counter() - t0
            frame = cv2.flip(frame, 1)  # 거울 모드 (실시간 시스템과 동일하게)
            lm, hand, score = det.detect(frame, now)

            label = labels[label_i]
            if recording:
                frame_idx += 1
                rec.write(
                    {
                        "participant": args.participant,
                        "session": args.session,
                        "condition": args.condition,
                        "trial_id": trial_id,
                        "label": label,
                        "frame_idx": frame_idx,
                        "t": round(now - trial_start, 4),
                        "width": w,
                        "height": h,
                        "hand_detected": int(lm is not None),
                        "handedness": hand,
                        "score": round(score, 4),
                    },
                    lm,
                )

            # ---- 화면 표시 ----
            guess = ""
            if lm is not None:
                draw_hand(frame, lm)
                r = classify(lm, hand, aspect)
                guess = f"rule: {r.gesture} ({r.confidence:.2f}) {''.join(map(str, r.fingers))}"
            color = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(frame, f"[{label_i + 1}] {label}  {'REC' if recording else 'ready'}  trial {trial_id}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, guess, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            keys = "  ".join(f"{i + 1}:{lab}({counts[lab]})" for i, lab in enumerate(labels))
            cv2.putText(frame, keys, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("collect (SPACE rec, 1-9 label, q quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord(str(len(labels))) and not recording:
                label_i = key - ord("1")
            if key == ord(" "):
                recording = not recording
                if recording:
                    trial_id += 1
                    frame_idx = 0
                    trial_start = now
                else:
                    counts[label] += 1
                    rec.flush()
                    print(f"trial {trial_id} 저장: {label}")

    rec.close()
    cap.release()
    cv2.destroyAllWindows()
    print("수집 요약:", counts)


if __name__ == "__main__":
    main()
