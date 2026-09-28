#!/usr/bin/env python3
"""
analyze_clips.py

Two things, controlled by CLI flags:
  1. --summary   : scan a folder of video clips and report clip-length /
                   category stats (fast, no ML deps beyond opencv).
  2. --autolabel : sample frames from one clip and zero-shot classify them
                   against a candidate action vocabulary using CLIP.
                   (Slower, downloads a pretrained model on first run.)

Usage:
    python analyze_clips.py --clips_dir ./clips --summary
    python analyze_clips.py --clips_dir ./clips --autolabel --clip clip_01.mp4
"""

import argparse
import os
import sys
import glob

import cv2


CANDIDATE_ACTIONS = [
    "picks up an object",
    "puts down an object",
    "opens a drawer",
    "closes a drawer",
    "opens a door",
    "closes a door",
    "pours liquid",
    "cuts food",
    "washes hands",
    "wipes a surface",
    "walks",
    "stands still",
]


def get_clip_duration(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0:
        return None
    return frame_count / fps


def summarize(clips_dir):
    paths = sorted(
        glob.glob(os.path.join(clips_dir, "*.mp4"))
        + glob.glob(os.path.join(clips_dir, "*.mov"))
        + glob.glob(os.path.join(clips_dir, "*.avi"))
    )
    if not paths:
        print(f"No video files found in {clips_dir}")
        return

    durations = []
    print(f"{'File':40s} {'Duration (s)':>12s}")
    print("-" * 54)
    for p in paths:
        dur = get_clip_duration(p)
        durations.append(dur)
        dur_str = f"{dur:.1f}" if dur is not None else "unreadable"
        print(f"{os.path.basename(p):40s} {dur_str:>12s}")

    valid = [d for d in durations if d is not None]
    if valid:
        print("-" * 54)
        print(f"Clips found:      {len(paths)}")
        print(f"Readable clips:   {len(valid)}")
        print(f"Total duration:   {sum(valid):.1f} s")
        print(f"Average duration: {sum(valid) / len(valid):.1f} s")
        print(f"Min / Max:        {min(valid):.1f} s / {max(valid):.1f} s")


def sample_frames(path, n=6):
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count == 0:
        cap.release()
        return []
    indices = [int(frame_count * i / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))
    cap.release()
    return frames


def autolabel(clips_dir, clip_name):
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        print(
            "Missing deps for --autolabel. Run: pip install torch transformers pillow",
            file=sys.stderr,
        )
        sys.exit(1)

    path = os.path.join(clips_dir, clip_name)
    if not os.path.exists(path):
        print(f"Clip not found: {path}", file=sys.stderr)
        sys.exit(1)

    print("Loading CLIP model (first run downloads weights)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    frames = sample_frames(path, n=6)
    if not frames:
        print("Could not read frames from clip.")
        return

    fps = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS) or 30

    print(f"\nZero-shot action guesses for {clip_name}:")
    print("-" * 60)
    for idx, frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        inputs = processor(
            text=CANDIDATE_ACTIONS, images=img, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        top_idx = probs.argmax().item()
        timestamp = idx / fps
        print(
            f"  t={timestamp:6.1f}s  -> {CANDIDATE_ACTIONS[top_idx]:25s} "
            f"(conf {probs[top_idx]:.2f})"
        )
    print("-" * 60)
    print(
        "Note: this is a rough per-frame zero-shot demo, not a production "
        "labeler — it has no temporal context and a fixed vocabulary."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", default="./clips")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--autolabel", action="store_true")
    ap.add_argument("--clip", help="filename of clip to autolabel (in clips_dir)")
    args = ap.parse_args()

    if not args.summary and not args.autolabel:
        args.summary = True  # default action

    if args.summary:
        summarize(args.clips_dir)

    if args.autolabel:
        if not args.clip:
            print("--autolabel requires --clip <filename>", file=sys.stderr)
            sys.exit(1)
        autolabel(args.clips_dir, args.clip)


if __name__ == "__main__":
    main()
