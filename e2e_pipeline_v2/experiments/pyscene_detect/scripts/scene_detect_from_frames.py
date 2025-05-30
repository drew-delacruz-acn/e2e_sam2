#!/usr/bin/env python3
"""scene_detect_from_frames.py
Detect scene boundaries in a folder of numbered image frames using PySceneDetect.

Example:
    python scene_detect_from_frames.py /path/to/frames --fps 30 --ext .jpg --detector adaptive --sigma 0.3 \
           --min-scene-len 15 --output scenes.csv

Positional arguments
--------------------
folder              The directory containing image frames named 0.jpg, 1.jpg … or 0001.png, …

Optional arguments
------------------
--fps FLOAT         Frames-per-second of the source sequence (REQUIRED for image sequences).
--ext EXT           File extension (default: .jpg). Must match your frames.
--detector NAME     adaptive | content | threshold  (default: adaptive)
--threshold FLOAT   Content/threshold detector cut-score (default: 27).
--sigma FLOAT       Adaptive detector rolling-avg factor (default: 0.33).
--min-scene-len N   Minimum scene length in frames (default: 15).
--output PATH       Path to write CSV summary (default: scenes.csv in the folder).
--skip-csv          Don't write a CSV – print to stdout only.
"""
import argparse
import csv
import sys
from pathlib import Path

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector

DEF_THRESHOLD = 27
DEF_SIGMA = 0.33
DEF_MIN_LEN = 15

def build_pattern(folder: Path, ext: str) -> str:
    """Return OpenCV-style pattern e.g. "/frames/%d.jpg" suitable for image sequences."""
    ext = ext if ext.startswith('.') else f'.{ext}'
    return str(folder / f"%d{ext}")

def choose_detector(name: str, thr: float, sigma: float, min_len: int):
    name = name.lower()
    if name == "adaptive":
        return AdaptiveDetector(min_scene_len=min_len)
    if name == "content":
        return ContentDetector(threshold=thr, min_scene_len=min_len)
    if name == "threshold":
        return ThresholdDetector(threshold=thr, min_scene_len=min_len)
    raise ValueError(f"Unknown detector: {name}")

def write_csv(scenes, csv_path: Path):
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "start_frame", "end_frame", "start_time_sec", "end_time_sec"])
        for idx, (start, end) in enumerate(scenes, 1):
            writer.writerow([
                idx,
                start.get_frames(),
                end.get_frames(),
                f"{start.get_seconds():.3f}",
                f"{end.get_seconds():.3f}",
            ])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect scene cuts in an image sequence folder.")
    parser.add_argument("folder", type=Path, help="Path to folder containing numbered frames.")
    parser.add_argument("--fps", type=float, required=True, help="Frames-per-second of the input sequence.")
    parser.add_argument("--ext", default=".jpg", help="Image extension (e.g. .jpg, .png). Defaults to .jpg")
    parser.add_argument("--detector", default="adaptive", choices=["adaptive", "content", "threshold"],
                        help="Scene detection algorithm.")
    parser.add_argument("--threshold", type=float, default=DEF_THRESHOLD,
                        help="Threshold for content/threshold detectors.")
    parser.add_argument("--sigma", type=float, default=DEF_SIGMA,
                        help="Sigma factor for adaptive detector (0.0–1.0).")
    parser.add_argument("--min-scene-len", type=int, default=DEF_MIN_LEN,
                        help="Merge scenes shorter than this many frames.")
    parser.add_argument("--output", type=Path, default=None, help="CSV output path (default <folder>/scenes.csv)")
    parser.add_argument("--skip-csv", action="store_true", help="Skip writing CSV – only print to stdout.")
    args = parser.parse_args(argv)

    if not args.folder.is_dir():
        parser.error(f"{args.folder} is not a directory")

    pattern = build_pattern(args.folder, args.ext)
    print(f"[INFO] Reading sequence pattern {pattern}")

    video = open_video(pattern, framerate=args.fps)

    scene_manager = SceneManager()
    detector = choose_detector(args.detector, args.threshold, args.sigma, args.min_scene_len)
    scene_manager.add_detector(detector)

    print("[INFO] Detecting scenes – this may take a while…")
    scene_manager.detect_scenes(video, show_progress=True)
    scenes = scene_manager.get_scene_list()

    if not scenes:
        print("[INFO] No scenes detected.")
        return

    for idx, (start, end) in enumerate(scenes, 1):
        print(f"Scene {idx:03d}: frames {start.get_frames():>6} – {end.get_frames():>6}"
              f" (time {start.get_timecode()} – {end.get_timecode()})")

    if not args.skip_csv:
        csv_path = args.output or (args.folder / "scenes.csv")
        write_csv(scenes, csv_path)
        print(f"[INFO] Scene list written to {csv_path}")

if __name__ == "__main__":
    main() 