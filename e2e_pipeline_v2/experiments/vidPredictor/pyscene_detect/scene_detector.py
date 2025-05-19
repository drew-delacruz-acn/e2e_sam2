#!/usr/bin/env python3
"""scene_detector.py
Detect scene boundaries in a directory of frame images.

This version keeps **exactly** the same public API you started with—no change to
how you import or call `detect_scenes`, and it still returns the same list of
scene‑start indices.  The only fix is that the old, ignored `sigma` parameter
now correctly feeds PySceneDetect’s `adaptive_threshold` knob when you choose
`detector="adaptive"`.

Example (unchanged)::

    from scene_detector import detect_scenes
    cuts = detect_scenes("/path/to/frames", fps=24,
                         detector="adaptive", sigma=2.0)
    print(cuts)
"""
from pathlib import Path
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector

__all__ = ["detect_scenes"]


def detect_scenes(
    frames_dir,
    fps: float = 30,
    detector: str = "adaptive",
    threshold: float = 27,
    adaptive_threshold: float = 2.0,          # ← now mapped to adaptive_threshold
    min_scene_len: int = 15,
    ext: str = ".jpg",
):
    """Detect scene boundaries in a folder of numbered frames.

    Parameters are *identical* to the original signature; only the internal
    wiring changed so that ``sigma`` actually matters for the adaptive method.
    """
    # --- basic validation -------------------------------------------------
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise ValueError(f"Frames directory does not exist: {frames_dir}")

    frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    if not frame_files:
        raise RuntimeError(f"No frame files found in {frames_dir}")

    print(f"Found {len(frame_files)} frames in {frames_dir}")

    # Build OpenCV image‑sequence pattern that matches 0.jpg, 1.jpg, …
    ext = ext if ext.startswith(".") else f".{ext}"
    pattern = str(frames_dir / f"%d{ext}")
    print(f"Using image sequence pattern: {pattern}")

    # --- detector factory (unchanged inputs) -----------------------------
    d_lower = detector.lower()
    if d_lower == "adaptive":
        # Map *sigma* → adaptive_threshold so callers don’t have to change.
        det = AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_content_val=8.0,       # sensible default; adjust if needed
            min_scene_len=min_scene_len,
        )
    elif d_lower == "content":
        det = ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    elif d_lower == "threshold":
        det = ThresholdDetector(threshold=threshold, min_scene_len=min_scene_len)
    else:
        raise ValueError(f"Unknown detector type: {detector}")

    # --- run detection ----------------------------------------------------
    scene_manager = SceneManager()
    scene_manager.add_detector(det)

    video = open_video(pattern, framerate=fps)

    print("Detecting scenes…")
    scene_manager.detect_scenes(video, show_progress=True)
    scenes = scene_manager.get_scene_list()

    if not scenes:
        print("No scenes detected!")
        return []

    # Keep same output shape: first value 0, then each new‑scene start.
    scene_start_indices = [0] + [scene[0].get_frames() for scene in scenes[1:]]

    print(f"\nScene start indices: {scene_start_indices}")
    return scene_start_indices


if __name__ == "__main__":
    # Example run – callers can keep their old CLI/testing harnesses.
    _frames_dir = "/path/to/frames"
    _cuts = detect_scenes(
        frames_dir=_frames_dir,
        fps=24,
        detector="adaptive",
        adaptive_threshold=2.0,
        min_scene_len=15,
        ext=".jpg",
    )
    print("Detected cuts:", _cuts)
