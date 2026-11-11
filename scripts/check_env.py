"""개발 환경 확인: 웹캠 + MediaPipe 손 랜드마커 + 규칙 판별기가 함께 동작하는지 본다.

실행:  python scripts/check_env.py [--camera 1] [--width 1280 --height 720] [--no-mjpg]
종료:  q 또는 ESC
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gesture.camera import HandDetector, draw_hand, open_camera  # noqa: E402
from src.gesture.rules import classify  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0, help="웹캠 번호 (외장 웹캠은 보통 1)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-mjpg", action="store_true", help="MJPG 압축 요청 끄기 (비교용)")
    args = p.parse_args()

    cap, w, h, fps = open_camera(args.camera, args.width, args.height, mjpg=not args.no_mjpg)
    print(f"웹캠 {args.camera}: {w}x{h}, 요청 FPS 응답 {fps:.0f}")

    with HandDetector() as det:
        t0 = prev = time.perf_counter()
        meas = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임을 읽지 못했습니다.")
                break
            now = time.perf_counter()
            frame = cv2.flip(frame, 1)

            s = time.perf_counter()
            lm, hand, score = det.detect(frame, now - t0)
            infer_ms = (time.perf_counter() - s) * 1000

            label = "no hand"
            if lm is not None:
                draw_hand(frame, lm)
                r = classify(lm, hand, w / h)
                label = f"{hand} {score:.2f} | {r.gesture} {''.join(map(str, r.fingers))}"

            meas = 0.9 * meas + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            cv2.putText(frame, f"FPS {meas:4.1f} | infer {infer_ms:4.1f} ms | {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("check_env (q: quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
