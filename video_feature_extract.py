import argparse
import glob
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# Optional dependency.
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    mp = None
    HAS_MEDIAPIPE = False

HAS_TRANSFORMERS = False


@dataclass
class VideoMetadata:
    file_path: str
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


@dataclass
class SegmentFeatures:
    video_file: str
    segment_id: str
    segmentation_mode: str
    start_sec: float
    end_sec: float
    duration_sec: float

    mean_motion_intensity: float
    std_motion_intensity: float
    max_motion_intensity: float
    motion_peak_count: int
    motion_peak_rate_hz: float
    motion_periodicity_score: float
    dominant_motion_freq_hz: float

    mean_flow_magnitude: float
    flow_rhythm_score: float
    dominant_flow_freq_hz: float

    cut_count: int
    cut_frequency_hz: float

    mean_camera_dx: float
    mean_camera_dy: float
    mean_camera_motion: float
    camera_motion_stability: float

    pose_detected_ratio: float
    pose_motion_mean: float
    pose_motion_std: float
    pose_trajectory_length_mean: float

    brightness_mean: float
    brightness_std: float
    saturation_mean: float
    contrast_mean: float
    colorfulness_mean: float

    semantic_tags: str = ""
    semantic_scores: str = ""
    style_tags: str = ""

    extra: Dict = field(default_factory=dict)


# -----------------------------
# Utilities
# -----------------------------

def find_mp4_files(input_path: str) -> List[str]:
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == ".mp4":
        return [str(path.resolve())]

    if path.is_dir():
        files = sorted(str(p.resolve()) for p in path.glob("*.mp4"))
        if files:
            return files

    # fallback: current folder if the path doesn't exist or is omitted strangely
    files = sorted(str(Path(".").resolve() / f) for f in glob.glob("*.mp4"))
    return files


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    if len(x) == 0:
        return x
    window = max(1, int(window))
    if window == 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if len(values) > 0 else 0.0


def safe_std(values: List[float]) -> float:
    return float(np.std(values)) if len(values) > 0 else 0.0


def safe_max(values: List[float]) -> float:
    return float(np.max(values)) if len(values) > 0 else 0.0


def robust_colorfulness(bgr_frame: np.ndarray) -> float:
    # Hasler-Susstrunk style colorfulness metric.
    frame = bgr_frame.astype(np.float32)
    (B, G, R) = cv2.split(frame)
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))


def estimate_periodicity(signal: np.ndarray, sample_rate_hz: float) -> Tuple[float, float]:
    """
    Returns:
        periodicity_score in [0, 1]-ish
        dominant_freq_hz
    """
    if len(signal) < 8 or sample_rate_hz <= 0:
        return 0.0, 0.0

    x = np.asarray(signal, dtype=np.float32)
    x = x - np.mean(x)
    if np.allclose(x, 0):
        return 0.0, 0.0

    x = moving_average(x, window=5)
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate_hz)
    power = np.abs(fft) ** 2

    # Ignore DC and implausibly high frequencies for human motion/cuts.
    valid = (freqs >= 0.1) & (freqs <= 5.0)
    if not np.any(valid):
        return 0.0, 0.0

    power_valid = power[valid]
    freqs_valid = freqs[valid]
    if len(power_valid) == 0 or np.sum(power_valid) <= 1e-8:
        return 0.0, 0.0

    idx = int(np.argmax(power_valid))
    dominant_freq = float(freqs_valid[idx])
    periodicity_score = float(power_valid[idx] / (np.sum(power_valid) + 1e-8))
    return periodicity_score, dominant_freq


def frame_diff_cut_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    diff = cv2.absdiff(prev_gray, gray)
    return float(np.mean(diff))


# -----------------------------
# Optional model wrappers
# -----------------------------

class PoseExtractor:
    def __init__(self):
        self.enabled = False
        self.pose = None

        if not HAS_MEDIAPIPE:
            print("[PoseExtractor] MediaPipe not installed. Pose features disabled.")
            return

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            try:
                self.pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.enabled = True
                print("[PoseExtractor] Using MediaPipe Solutions API.")
                return
            except Exception as e:
                print(f"[PoseExtractor] Failed to initialize Solutions API: {e}")

        print("[PoseExtractor] MediaPipe pose API not available. Pose features disabled.")

    def close(self):
        if self.pose is not None:
            self.pose.close()

    def extract_pose_vector(self, bgr_frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.enabled or self.pose is None:
            return None
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None
        coords = []
        for lm in result.pose_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)


# -----------------------------
# Video analysis core
# -----------------------------

def read_video_metadata(video_path: str) -> VideoMetadata:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0:
        fps = 25.0
    duration_sec = frame_count / fps if fps > 0 else 0.0

    return VideoMetadata(
        file_path=video_path,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        width=width,
        height=height,
    )


def build_segments(duration_sec: float, mode: str, step_sec: Optional[float],
                   segment_sec: float = 2.0, beat_duration_sec: Optional[float] = None,
                   bar_duration_sec: Optional[float] = None) -> List[Tuple[float, float]]:
    if mode == "seconds":
        win = segment_sec
    elif mode == "4beats":
        if beat_duration_sec is None:
            raise ValueError("4beats mode requires --beat-duration-sec or --bpm")
        win = 4.0 * beat_duration_sec
    elif mode == "1bar":
        if bar_duration_sec is None:
            raise ValueError("1bar mode requires --bar-duration-sec or --bpm")
        win = bar_duration_sec
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    step = step_sec if step_sec is not None else win
    step = max(step, 0.1)
    win = max(win, 0.1)

    segments = []
    start = 0.0
    while start < duration_sec:
        end = min(start + win, duration_sec)
        if end - start >= min(0.5, win):
            segments.append((round(start, 4), round(end, 4)))
        if end >= duration_sec:
            break
        start += step
    return segments


def sample_frame_indices(start_frame: int, end_frame: int, max_samples: int) -> List[int]:
    count = max(0, end_frame - start_frame)
    if count <= 0:
        return []
    if count <= max_samples:
        return list(range(start_frame, end_frame))
    idxs = np.linspace(start_frame, end_frame - 1, num=max_samples, dtype=int)
    return list(np.unique(idxs))


def estimate_camera_motion(prev_gray: np.ndarray, gray: np.ndarray) -> Tuple[float, float, float]:
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=8)
    if prev_pts is None:
        return 0.0, 0.0, 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
    if next_pts is None or status is None:
        return 0.0, 0.0, 0.0

    good_prev = prev_pts[status.flatten() == 1]
    good_next = next_pts[status.flatten() == 1]
    if len(good_prev) < 8:
        return 0.0, 0.0, 0.0

    shifts = good_next - good_prev
    dx = float(np.median(shifts[:, 0, 0]))
    dy = float(np.median(shifts[:, 0, 1]))
    mag = float(np.sqrt(dx * dx + dy * dy))
    return dx, dy, mag


def compute_flow_features(prev_gray: np.ndarray, gray: np.ndarray) -> Tuple[float, float]:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = float(np.mean(mag))
    std_mag = float(np.std(mag))
    return mean_mag, std_mag


def analyze_segment(
    video_path: str,
    metadata: VideoMetadata,
    start_sec: float,
    end_sec: float,
    segmentation_mode: str,
    pose_extractor: PoseExtractor,
    sample_fps: float = 6.0,
    cut_threshold: float = 22.0,
) -> SegmentFeatures:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    start_frame = int(start_sec * metadata.fps)
    end_frame = min(int(math.ceil(end_sec * metadata.fps)), metadata.frame_count)
    segment_duration = max(0.0, end_sec - start_sec)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_stride = max(1, int(round(metadata.fps / sample_fps)))

    prev_gray = None
    prev_pose = None

    motion_series = []
    flow_series = []
    camera_series = []
    camera_dx = []
    camera_dy = []
    cut_scores = []
    cut_count = 0

    brightness_values = []
    saturation_values = []
    contrast_values = []
    colorfulness_values = []

    pose_motion_values = []
    pose_detected_frames = 0
    pose_traj_lengths = []


    frame_idx = start_frame
    processed_count = 0

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_idx - start_frame) % frame_stride != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        brightness_values.append(float(np.mean(gray)))
        contrast_values.append(float(np.std(gray)))
        saturation_values.append(float(np.mean(hsv[..., 1])))
        colorfulness_values.append(robust_colorfulness(frame))

        if prev_gray is not None:
            cut_score = frame_diff_cut_score(prev_gray, gray)
            cut_scores.append(cut_score)
            if cut_score > cut_threshold:
                cut_count += 1

            mean_mag, _ = compute_flow_features(prev_gray, gray)
            flow_series.append(mean_mag)
            motion_series.append(mean_mag)

            dx, dy, cam_mag = estimate_camera_motion(prev_gray, gray)
            camera_dx.append(dx)
            camera_dy.append(dy)
            camera_series.append(cam_mag)

        pose_vec = pose_extractor.extract_pose_vector(frame)
        if pose_vec is not None:
            pose_detected_frames += 1
            if prev_pose is not None and len(prev_pose) == len(pose_vec):
                pose_delta = pose_vec - prev_pose
                pose_motion = float(np.linalg.norm(pose_delta) / len(pose_vec))
                pose_motion_values.append(pose_motion)
                pose_traj_lengths.append(float(np.linalg.norm(pose_delta)))
            prev_pose = pose_vec

        prev_gray = gray
        processed_count += 1
        frame_idx += 1

    cap.release()

    motion_arr = np.array(motion_series, dtype=np.float32)
    flow_arr = np.array(flow_series, dtype=np.float32)
    cam_arr = np.array(camera_series, dtype=np.float32)

    motion_sample_rate = sample_fps
    motion_periodicity_score, dominant_motion_freq_hz = estimate_periodicity(motion_arr, motion_sample_rate)
    flow_rhythm_score, dominant_flow_freq_hz = estimate_periodicity(flow_arr, motion_sample_rate)

    peaks, _ = find_peaks(motion_arr, distance=max(1, int(sample_fps * 0.2))) if len(motion_arr) > 3 else ([], {})
    motion_peak_count = int(len(peaks))
    motion_peak_rate_hz = float(motion_peak_count / max(segment_duration, 1e-8))

    return SegmentFeatures(
        video_file=os.path.basename(video_path),
        segment_id=f"{Path(video_path).stem}_{segmentation_mode}_{start_sec:.2f}_{end_sec:.2f}",
        segmentation_mode=segmentation_mode,
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        duration_sec=float(segment_duration),

        mean_motion_intensity=safe_mean(motion_series),
        std_motion_intensity=safe_std(motion_series),
        max_motion_intensity=safe_max(motion_series),
        motion_peak_count=motion_peak_count,
        motion_peak_rate_hz=motion_peak_rate_hz,
        motion_periodicity_score=float(motion_periodicity_score),
        dominant_motion_freq_hz=float(dominant_motion_freq_hz),

        mean_flow_magnitude=safe_mean(flow_series),
        flow_rhythm_score=float(flow_rhythm_score),
        dominant_flow_freq_hz=float(dominant_flow_freq_hz),

        cut_count=int(cut_count),
        cut_frequency_hz=float(cut_count / max(segment_duration, 1e-8)),

        mean_camera_dx=safe_mean(camera_dx),
        mean_camera_dy=safe_mean(camera_dy),
        mean_camera_motion=safe_mean(camera_series),
        camera_motion_stability=float(1.0 / (1.0 + safe_std(camera_series))),

        pose_detected_ratio=float(pose_detected_frames / max(processed_count, 1)),
        pose_motion_mean=safe_mean(pose_motion_values),
        pose_motion_std=safe_std(pose_motion_values),
        pose_trajectory_length_mean=safe_mean(pose_traj_lengths),

        brightness_mean=safe_mean(brightness_values),
        brightness_std=safe_std(brightness_values),
        saturation_mean=safe_mean(saturation_values),
        contrast_mean=safe_mean(contrast_values),
        colorfulness_mean=safe_mean(colorfulness_values),

        semantic_tags="",
        semantic_scores="{}",
        style_tags="",
        extra={
            "processed_frames": processed_count,
            "mean_cut_score": safe_mean(cut_scores),
        },
    )


# -----------------------------
# Output helpers
# -----------------------------

def save_results(features: List[SegmentFeatures], out_dir: str, stem: str = "video_features") -> None:
    ensure_dir(out_dir)

    records = []
    for f in features:
        d = asdict(f)
        d["extra"] = json.dumps(d.get("extra", {}))
        records.append(d)

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    json_path = os.path.join(out_dir, f"{stem}.json")
    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2, ensure_ascii=False)

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


def save_video_summary(video_summaries: List[Dict], out_dir: str, stem: str = "video_summary") -> None:
    ensure_dir(out_dir)
    json_path = os.path.join(out_dir, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(video_summaries, fp, indent=2, ensure_ascii=False)
    print(f"Saved summary: {json_path}")


# -----------------------------
# Main driver
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract video segment features from one or many MP4 files in a folder."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=".",
        help="Path to a single .mp4 file or a folder containing .mp4 files (default: current folder).",
    )
    parser.add_argument(
        "--segment-mode",
        choices=["seconds", "4beats", "1bar", "all"],
        default="all",
        help="Segmentation mode. 'all' runs seconds + 4beats + 1bar when durations are available.",
    )
    parser.add_argument("--segment-sec", type=float, default=2.0, help="Window length for seconds mode.")
    parser.add_argument("--step-sec", type=float, default=1.0, help="Step size for overlapping windows.")

    parser.add_argument("--bpm", type=float, default=None, help="Optional song BPM, used to derive beat/bar durations.")
    parser.add_argument("--beat-duration-sec", type=float, default=None, help="Optional beat duration in seconds.")
    parser.add_argument("--bar-duration-sec", type=float, default=None, help="Optional bar duration in seconds.")

    parser.add_argument("--sample-fps", type=float, default=6.0, help="Analysis sampling FPS inside each segment.")
    parser.add_argument("--cut-threshold", type=float, default=22.0, help="Frame-difference threshold for cut detection.")
    parser.add_argument("--out-dir", default="video_features_out", help="Output directory.")
    args = parser.parse_args()

    files = find_mp4_files(args.input_path)
    if not files:
        raise FileNotFoundError("No .mp4 files found in the given path or current folder.")

    beat_duration_sec = args.beat_duration_sec
    bar_duration_sec = args.bar_duration_sec
    if args.bpm is not None and args.bpm > 0:
        beat_duration_sec = 60.0 / args.bpm
        bar_duration_sec = 4.0 * beat_duration_sec
    if beat_duration_sec is not None and bar_duration_sec is None:
        bar_duration_sec = 4.0 * beat_duration_sec

    modes = [args.segment_mode] if args.segment_mode != "all" else ["seconds"]
    if args.segment_mode == "all":
        if beat_duration_sec is not None:
            modes.append("4beats")
        if bar_duration_sec is not None:
            modes.append("1bar")

    print("Found videos:")
    for f in files:
        print(f"  - {f}")
    print(f"Segmentation modes: {modes}")

    pose_extractor = PoseExtractor()

    all_features: List[SegmentFeatures] = []
    summaries: List[Dict] = []

    try:
        for video_path in files:
            metadata = read_video_metadata(video_path)
            print("\n" + "=" * 80)
            print(f"Processing: {video_path}")
            print(f"Duration: {metadata.duration_sec:.2f}s | FPS: {metadata.fps:.3f} | Size: {metadata.width}x{metadata.height}")

            video_segment_count = 0
            for mode in modes:
                segments = build_segments(
                    duration_sec=metadata.duration_sec,
                    mode=mode,
                    step_sec=args.step_sec,
                    segment_sec=args.segment_sec,
                    beat_duration_sec=beat_duration_sec,
                    bar_duration_sec=bar_duration_sec,
                )
                print(f"  {mode}: {len(segments)} segments")

                for start_sec, end_sec in segments:
                    features = analyze_segment(
                        video_path=video_path,
                        metadata=metadata,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        segmentation_mode=mode,
                        pose_extractor=pose_extractor,
                        sample_fps=args.sample_fps,
                        cut_threshold=args.cut_threshold,
                    )
                    all_features.append(features)
                    video_segment_count += 1

            summaries.append(
                {
                    "video_file": os.path.basename(video_path),
                    "file_path": video_path,
                    "duration_sec": metadata.duration_sec,
                    "fps": metadata.fps,
                    "frame_count": metadata.frame_count,
                    "width": metadata.width,
                    "height": metadata.height,
                    "segments_processed": video_segment_count,
                }
            )

        save_results(all_features, args.out_dir, stem="video_features")
        save_video_summary(summaries, args.out_dir, stem="video_summary")

        print("\nDone.")
        print(f"Total segments processed: {len(all_features)}")
        print(f"MediaPipe available: {HAS_MEDIAPIPE}")

    finally:
        pose_extractor.close()


if __name__ == "__main__":
    main()
