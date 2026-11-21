"""가짜 수집 데이터 생성 — 웹캠 없이 학습·평가 스크립트를 처음부터 끝까지 시험하기 위한 것.

실제 성능과는 무관하다. 결과 숫자를 보고서에 쓰지 말 것.

실행: python scripts/make_synthetic_data.py   → data/synthetic/*.csv
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gesture.dataio import CsvRecorder  # noqa: E402
from src.gesture.synthetic import make_hand  # noqa: E402

GESTURES = ["palm", "fist", "peace", "call", "rock", "ok", "like", "one", "none"]
NONOP = {  # 비조작 동작을 손 모양 흐름으로 흉내
    "nonop_drink": ["fist", "fist", "none_like"],
    "nonop_head": ["palm", "one", "palm"],
    "nonop_wave": ["palm", "palm", "palm"],
    "nonop_phone": ["fist", "like", "fist"],
    "nonop_point": ["one", "one", "one"],
}
FPS = 30
W, H = 1280, 720


def frames_for(gesture, n, rng, hand):
    for _ in range(n):
        g = gesture
        if g == "none_like":  # 손이 잠깐 안 보임
            yield None
            continue
        yield make_hand(
            g,
            center=(rng.uniform(0.3, 0.7), rng.uniform(0.4, 0.7)),
            size=rng.uniform(0.06, 0.12),
            handedness=hand,
            aspect=W / H,
            rotation_deg=rng.uniform(-20, 20),
            noise=0.03,
            rng=rng,
        )


def write_trial(rec, meta, frames, hand):
    for i, lm in enumerate(frames):
        rec.write(
            {**meta, "frame_idx": i + 1, "t": round(i / FPS, 4), "width": W, "height": H,
             "hand_detected": int(lm is not None), "handedness": hand if lm is not None else "",
             "score": 0.95 if lm is not None else 0.0},
            lm,
        )


def main(out_dir=ROOT / "data" / "synthetic", n_participants=5, seed=0):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    for f in out_dir.glob("*.csv"):
        f.unlink()
    for p in range(1, n_participants + 1):
        pid = f"S{p:02d}"
        hand = "Right" if p % 3 else "Left"
        rec = CsvRecorder(out_dir / f"{pid}_synthetic.csv")
        trial = 0
        for g in GESTURES:  # 제스처: 2회 × 1초
            for _ in range(2):
                trial += 1
                meta = {"participant": pid, "session": "s1", "condition": "bright" if trial % 2 else "dim",
                        "trial_id": trial, "label": g}
                write_trial(rec, meta, list(frames_for(g, FPS, rng, hand)), hand)
        for lab, seq in NONOP.items():  # 비조작: 4회 × 3구간 × 0.7초
            for _ in range(4):
                trial += 1
                frames = [f for g in seq for f in frames_for(g, 21, rng, hand)]
                write_trial(rec, {"participant": pid, "session": "s1", "condition": "", "trial_id": trial,
                                  "label": lab}, frames, hand)
        for cand in ("call", "rock", "ok"):  # 클러치 후보별 의도적 활성화: 손바닥 0.5초 → 후보 1초
            for _ in range(10):
                trial += 1
                frames = list(frames_for("palm", 15, rng, hand)) + list(frames_for(cand, 30, rng, hand))
                write_trial(rec, {"participant": pid, "session": "s1", "condition": "", "trial_id": trial,
                                  "label": f"activate_{cand}"}, frames, hand)
        for _ in range(10):  # 베이스라인(단순 활성화) 시도: 손 내린 상태 0.5초 → 손바닥 1초
            trial += 1
            frames = list(frames_for("none", 15, rng, hand)) + list(frames_for("palm", 30, rng, hand))
            write_trial(rec, {"participant": pid, "session": "s1", "condition": "", "trial_id": trial,
                              "label": "activate_palm"}, frames, hand)
        rec.close()
    print(f"가짜 데이터 {n_participants}명 → {out_dir}")


if __name__ == "__main__":
    main()
