"""실시간 데모: 웹캠 → 손 인식 → 제스처 판별 → 클러치 상태 머신 → /hand/state 화면 표시.

용도: (1) 규칙 임계값 튜닝 (오른쪽 막대에서 빨간 선 = 임계값)  (2) 팀 시연  (3) 지연 시간 확인

사용 예)
  python scripts/live_demo.py                       # 내장 웹캠, 규칙 기반, 클러치 call
  python scripts/live_demo.py --camera 1            # ZIO-C960
  python scripts/live_demo.py --clutch rock --log results/live_log.csv
  python scripts/live_demo.py --model models/gesture_clf.joblib   # 학습한 분류기로 판별
  python scripts/live_demo.py --record recordings/T01_A_basic.mp4  # 오버레이 화면 녹화 (얼굴 포함 → git 제외 폴더)

조작
  q/ESC  종료          r  상태 초기화 (IDLE)
  c      클러치 후보 전환 (call → rock → ok → palm)
  d      손가락 상세 막대 켜기/끄기
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gesture.camera import HandDetector, open_camera  # noqa: E402
from src.gesture.live import CLUTCH_CHOICES, LiveSession, draw_overlay, free_output_paths, state_row  # noqa: E402
from src.gesture.pipeline import GestureRecognizer  # noqa: E402


def load_classifier(path):
    import joblib

    from src.gesture.training import for_realtime

    return for_realtime(joblib.load(path)["model"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0, help="웹캠 번호 (ZIO-C960 = 1)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--model", default=None, help="학습 모델 경로 (없으면 규칙 기반)")
    p.add_argument("--clutch", default="call", choices=CLUTCH_CHOICES)
    p.add_argument("--log", default=None, help="프레임별 /hand/state를 CSV로 저장할 경로")
    p.add_argument("--no-mirror", action="store_true", help="좌우 반전 끄기")
    p.add_argument("--record", default=None, help="오버레이 화면을 mp4로 저장할 경로")
    args = p.parse_args()

    # 같은 이름으로 다시 실행해도 이전 녹화를 덮어쓰지 않게 번호를 붙인다 (영상·CSV 같은 번호)
    record, log = free_output_paths(args.record, args.log)
    if (record, log) != (args.record, args.log):
        print(f"이미 있는 파일이라 새 이름으로 저장합니다: {record or ''} {log or ''}")
    args.record, args.log = record, log

    cap, w, h, cam_fps = open_camera(args.camera, args.width, args.height)
    print(f"웹캠 {args.camera}: {w}x{h} @ {cam_fps:.0f}fps, 판별: {'분류기' if args.model else '규칙'}")

    clf = load_classifier(args.model) if args.model else None
    recognizer = GestureRecognizer(aspect=w / h, classifier=clf)

    log_f = writer = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.log, "w", newline="", encoding="utf-8")

    video, rec_fps, written = None, 30.0, 0
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        video = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), rec_fps, (w, h))
        if not video.isOpened():
            raise RuntimeError(f"녹화 파일을 만들 수 없습니다: {args.record}")

    detail, fps = True, 0.0
    t0 = last = time.perf_counter()
    try:
        with HandDetector() as det:
            session = LiveSession(det, recognizer, args.clutch)
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("프레임을 읽지 못했습니다.")
                    break
                now = time.perf_counter()
                dt, last = now - last, now
                fps = (1 / dt) if fps == 0 else 0.9 * fps + 0.1 * (1 / max(dt, 1e-6))  # 지수 평균
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)  # 거울 모드 (수집 스크립트와 동일)

                r = session.step(frame, now - t0)

                if log_f:
                    row = state_row(r, session.clutch)
                    if writer is None:
                        writer = csv.DictWriter(log_f, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)

                draw_overlay(frame, r, recognizer.rule_cfg, session.clutch, fps, detail)
                if video is not None:
                    # 실제 FPS가 30보다 낮아도 영상이 실제 시간대로 재생되게, 밀린 만큼 같은 프레임을 반복 저장
                    target = int((now - t0) * rec_fps) + 1
                    while written < target:
                        video.write(frame)
                        written += 1
                    cv2.circle(frame, (w - 25, h - 25), 10, (0, 0, 255), -1)  # 녹화 중 표시 (영상엔 안 들어감)
                cv2.imshow("live demo (q quit)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    session.reset()
                if key == ord("c"):
                    print("클러치 →", session.next_clutch())
                if key == ord("d"):
                    detail = not detail
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if log_f:
            log_f.close()
            print("로그 저장:", args.log)
        if video is not None:
            video.release()
            print(f"녹화 저장: {args.record} ({written / rec_fps:.1f}초)")


if __name__ == "__main__":
    main()
