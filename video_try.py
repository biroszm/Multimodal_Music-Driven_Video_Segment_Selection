import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ==========================================================
# SETTINGS
# ==========================================================
TRAINING_CSV = "segment_label_analysis_out/labeled_segments.csv"
INPUT_VIDEO = "vide_try.mp4"
OUT_DIR = "auto_label_out"
SEGMENT_SEC = 2.0
STEP_SEC = 1.0
SAMPLE_FPS = 4.0
TARGET_HEIGHT = 480
CUT_THRESHOLD = 22.0
PROBABILITY_THRESHOLD = 0.55
TOP_K = 30

# These are the most promising features from your earlier analysis.
CORE_FEATURES = [
    "mean_camera_motion",
    "camera_motion_stability",
    "brightness_std",
    "cut_frequency_hz",
    "motion_peak_rate_hz",
    "cut_count",
    "mean_camera_dy",
    "contrast_mean",
    "mean_motion_intensity",
    "mean_flow_magnitude",
]


# ==========================================================
# Utility helpers
# ==========================================================
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)



def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0



def safe_std(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0



def safe_max(values: List[float]) -> float:
    return float(np.max(values)) if values else 0.0



def resize_to_target_height(frame: np.ndarray, target_height: int = 480) -> np.ndarray:
    if target_height is None or target_height <= 0:
        return frame
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0 or h == target_height:
        return frame
    scale = target_height / float(h)
    target_width = max(1, int(round(w * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)



def robust_colorfulness(bgr_frame: np.ndarray) -> float:
    frame = bgr_frame.astype(np.float32)
    (B, G, R) = cv2.split(frame)
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))



def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    if len(x) == 0:
        return x
    window = max(1, int(window))
    if window == 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")



def estimate_periodicity(signal: np.ndarray, sample_rate_hz: float) -> Tuple[float, float]:
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


# ==========================================================
# Feature extraction on the new video
# ==========================================================
def build_segments(duration_sec: float, segment_sec: float, step_sec: float) -> List[Tuple[float, float]]:
    segments = []
    start = 0.0
    while start < duration_sec:
        end = min(start + segment_sec, duration_sec)
        if end - start >= min(0.5, segment_sec):
            segments.append((round(start, 4), round(end, 4)))
        if end >= duration_sec:
            break
        start += step_sec
    return segments



def read_video_metadata(video_path: str) -> Dict:
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

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }



def extract_segment_features(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float,
    frame_count: int,
    sample_fps: float,
    target_height: int,
    cut_threshold: float,
) -> Dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    start_frame = int(start_sec * fps)
    end_frame = min(int(math.ceil(end_sec * fps)), frame_count)
    duration_sec = max(0.0, end_sec - start_sec)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_stride = max(1, int(round(fps / sample_fps)))

    prev_gray = None
    motion_series = []
    flow_series = []
    camera_dx = []
    camera_dy = []
    camera_series = []
    brightness_values = []
    saturation_values = []
    contrast_values = []
    colorfulness_values = []
    cut_scores = []
    cut_count = 0

    frame_idx = start_frame
    processed_count = 0

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_to_target_height(frame, target_height=target_height)

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

        prev_gray = gray
        processed_count += 1
        frame_idx += 1

    cap.release()

    motion_arr = np.array(motion_series, dtype=np.float32)
    flow_arr = np.array(flow_series, dtype=np.float32)
    motion_periodicity_score, dominant_motion_freq_hz = estimate_periodicity(motion_arr, sample_fps)
    flow_rhythm_score, dominant_flow_freq_hz = estimate_periodicity(flow_arr, sample_fps)

    peaks, _ = find_peaks(motion_arr, distance=max(1, int(sample_fps * 0.2))) if len(motion_arr) > 3 else ([], {})
    motion_peak_count = int(len(peaks))
    motion_peak_rate_hz = float(motion_peak_count / max(duration_sec, 1e-8))

    return {
        "video_file": Path(video_path).name,
        "segment_id": f"{Path(video_path).stem}_{start_sec:.2f}_{end_sec:.2f}",
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "duration_sec": float(duration_sec),
        "mean_motion_intensity": safe_mean(motion_series),
        "std_motion_intensity": safe_std(motion_series),
        "max_motion_intensity": safe_max(motion_series),
        "motion_peak_count": motion_peak_count,
        "motion_peak_rate_hz": motion_peak_rate_hz,
        "motion_periodicity_score": float(motion_periodicity_score),
        "dominant_motion_freq_hz": float(dominant_motion_freq_hz),
        "mean_flow_magnitude": safe_mean(flow_series),
        "flow_rhythm_score": float(flow_rhythm_score),
        "dominant_flow_freq_hz": float(dominant_flow_freq_hz),
        "cut_count": int(cut_count),
        "cut_frequency_hz": float(cut_count / max(duration_sec, 1e-8)),
        "mean_camera_dx": safe_mean(camera_dx),
        "mean_camera_dy": safe_mean(camera_dy),
        "mean_camera_motion": safe_mean(camera_series),
        "camera_motion_stability": float(1.0 / (1.0 + safe_std(camera_series))),
        "pose_detected_ratio": 0.0,
        "pose_motion_mean": 0.0,
        "pose_motion_std": 0.0,
        "pose_trajectory_length_mean": 0.0,
        "brightness_mean": safe_mean(brightness_values),
        "brightness_std": safe_std(brightness_values),
        "saturation_mean": safe_mean(saturation_values),
        "contrast_mean": safe_mean(contrast_values),
        "colorfulness_mean": safe_mean(colorfulness_values),
        "processed_frames": processed_count,
        "mean_cut_score": safe_mean(cut_scores),
    }



def extract_features_for_new_video(
    video_path: str,
    segment_sec: float,
    step_sec: float,
    sample_fps: float,
    target_height: int,
    cut_threshold: float,
) -> pd.DataFrame:
    meta = read_video_metadata(video_path)
    segments = build_segments(meta["duration_sec"], segment_sec=segment_sec, step_sec=step_sec)

    print(f"Processing new video: {video_path}")
    print(f"Duration: {meta['duration_sec']:.2f}s | FPS: {meta['fps']:.3f} | Segments: {len(segments)}")

    rows = []
    for i, (start_sec, end_sec) in enumerate(segments, start=1):
        if i % 20 == 0 or i == len(segments):
            print(f"  Segment {i}/{len(segments)}")
        rows.append(
            extract_segment_features(
                video_path=video_path,
                start_sec=start_sec,
                end_sec=end_sec,
                fps=meta["fps"],
                frame_count=meta["frame_count"],
                sample_fps=sample_fps,
                target_height=target_height,
                cut_threshold=cut_threshold,
            )
        )
    return pd.DataFrame(rows)


# ==========================================================
# Training and prediction
# ==========================================================
def load_training_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "interesting_label" not in df.columns:
        raise ValueError("Training CSV must contain 'interesting_label'. Use your labeled_segments.csv file.")
    return df



def get_available_features(train_df: pd.DataFrame, predict_df: pd.DataFrame, preferred: List[str]) -> List[str]:
    return [f for f in preferred if f in train_df.columns and f in predict_df.columns]



def train_models(train_df: pd.DataFrame, feature_cols: List[str]):
    X = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = train_df["interesting_label"].astype(int)

    if y.nunique() < 2:
        raise ValueError("Training data needs both interesting and non-interesting labels.")

    logreg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    logreg.fit(X, y)

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        )),
    ])
    rf.fit(X, y)

    return logreg, rf



def compute_handcrafted_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    # Lower is better for these, so subtract z-scores.
    for col, weight in [
        ("mean_camera_motion", 1.2),
        ("brightness_std", 0.8),
        ("contrast_mean", 0.4),
    ]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            z = (s - s.mean()) / (s.std() + 1e-8)
            score += -weight * z.fillna(0.0)

    # Higher is better for these.
    for col, weight in [
        ("camera_motion_stability", 1.2),
        ("cut_frequency_hz", 0.6),
        ("motion_peak_rate_hz", 0.5),
        ("mean_motion_intensity", 0.25),
    ]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            z = (s - s.mean()) / (s.std() + 1e-8)
            score += weight * z.fillna(0.0)

    return score



def auto_label_segments(
    predict_df: pd.DataFrame,
    logreg,
    rf,
    feature_cols: List[str],
    probability_threshold: float,
) -> pd.DataFrame:
    df = predict_df.copy()
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    df["logreg_probability"] = logreg.predict_proba(X)[:, 1]
    df["rf_probability"] = rf.predict_proba(X)[:, 1]
    df["handcrafted_score"] = compute_handcrafted_score(df)

    # Normalize handcrafted score to 0..1
    hs = df["handcrafted_score"]
    hs_norm = (hs - hs.min()) / (hs.max() - hs.min() + 1e-8)
    df["handcrafted_probability"] = hs_norm

    df["combined_probability"] = (
        0.20 * df["logreg_probability"]
        + 0.50 * df["rf_probability"]
        + 0.30 * df["handcrafted_probability"]
    )
    df["auto_label"] = (df["combined_probability"] >= probability_threshold).astype(int)

    df = df.sort_values(["combined_probability", "start_sec"], ascending=[False, True]).reset_index(drop=True)
    return df


# ==========================================================
# Output and plotting
# ==========================================================
def save_outputs(df: pd.DataFrame, out_dir: str, top_k: int) -> None:
    ensure_dir(out_dir)
    all_csv = Path(out_dir) / "vide_try_auto_labels.csv"
    top_csv = Path(out_dir) / "vide_try_top_segments.csv"
    json_path = Path(out_dir) / "vide_try_auto_labels.json"

    df.to_csv(all_csv, index=False)
    df.head(top_k).to_csv(top_csv, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    print(f"Saved full results: {all_csv}")
    print(f"Saved top segments: {top_csv}")
    print(f"Saved JSON:         {json_path}")



def plot_results(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)

    plot_df = df.sort_values("start_sec").reset_index(drop=True)
    x_mid = (plot_df["start_sec"].values + plot_df["end_sec"].values) / 2.0
    auto = plot_df[plot_df["auto_label"] == 1]
    x_auto = (auto["start_sec"].values + auto["end_sec"].values) / 2.0

    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True)

    axes[0].plot(x_mid, plot_df["combined_probability"], linewidth=1.6, label="combined_probability")
    axes[0].scatter(x_auto, auto["combined_probability"], s=40, label="auto-labeled interesting")
    axes[0].axhline(PROBABILITY_THRESHOLD, linestyle="--", linewidth=1.0, alpha=0.8, label="threshold")
    axes[0].set_ylabel("Probability")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(x_mid, plot_df["mean_camera_motion"], linewidth=1.2, label="mean_camera_motion")
    axes[1].scatter(x_auto, auto["mean_camera_motion"], s=35, label="auto-labeled")
    axes[1].set_ylabel("Camera motion")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].plot(x_mid, plot_df["camera_motion_stability"], linewidth=1.2, label="camera_motion_stability")
    axes[2].scatter(x_auto, auto["camera_motion_stability"], s=35, label="auto-labeled")
    axes[2].set_ylabel("Stability")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right")

    axes[3].plot(x_mid, plot_df["brightness_std"], linewidth=1.2, label="brightness_std")
    axes[3].scatter(x_auto, auto["brightness_std"], s=35, label="auto-labeled")
    axes[3].set_ylabel("Brightness std")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].grid(alpha=0.25)
    axes[3].legend(loc="upper right")

    plt.suptitle("Automatic labeling of interesting segments in vide_try.mp4", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = Path(out_dir) / "vide_try_auto_label_plot.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot:         {png_path}")
    plt.show()



def print_summary(df: pd.DataFrame, top_k: int) -> None:
    df_time = df.sort_values("start_sec").reset_index(drop=True)
    auto_count = int(df_time["auto_label"].sum())
    print("=" * 90)
    print(f"Total segments analyzed: {len(df_time)}")
    print(f"Auto-labeled interesting segments: {auto_count}")
    print(f"Auto-label ratio: {auto_count / max(len(df_time), 1):.3f}")
    print("\nTop segments by combined probability:")
    show_cols = [
        "start_sec", "end_sec", "combined_probability", "rf_probability", "logreg_probability",
        "handcrafted_probability", "mean_camera_motion", "camera_motion_stability",
        "brightness_std", "cut_frequency_hz", "motion_peak_rate_hz"
    ]
    cols = [c for c in show_cols if c in df.columns]
    print(df.head(top_k)[cols].to_string(index=False))
    print("=" * 90)


# ==========================================================
# Main
# ==========================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-label interesting segments in vide_try.mp4 using your earlier labeled data and the best-performing features."
    )
    parser.add_argument("--video-path", default=INPUT_VIDEO, help="Path to the new video file.")
    parser.add_argument("--training-csv", default=TRAINING_CSV, help="Path to labeled_segments.csv from earlier manual analysis.")
    parser.add_argument("--segment-sec", type=float, default=SEGMENT_SEC, help="Segment duration in seconds.")
    parser.add_argument("--step-sec", type=float, default=STEP_SEC, help="Step between segments in seconds.")
    parser.add_argument("--sample-fps", type=float, default=SAMPLE_FPS, help="Sampling FPS inside each segment.")
    parser.add_argument("--target-height", type=int, default=TARGET_HEIGHT, help="Resize analyzed frames to this height.")
    parser.add_argument("--cut-threshold", type=float, default=CUT_THRESHOLD, help="Cut detection threshold.")
    parser.add_argument("--threshold", type=float, default=PROBABILITY_THRESHOLD, help="Probability threshold for auto-labeling.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="How many top segments to print/save separately.")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory.")
    args = parser.parse_args()

    train_df = load_training_data(args.training_csv)
    predict_df = extract_features_for_new_video(
        video_path=args.video_path,
        segment_sec=args.segment_sec,
        step_sec=args.step_sec,
        sample_fps=args.sample_fps,
        target_height=args.target_height,
        cut_threshold=args.cut_threshold,
    )

    feature_cols = get_available_features(train_df, predict_df, CORE_FEATURES)
    if len(feature_cols) < 4:
        raise ValueError(f"Not enough shared features found. Available: {feature_cols}")

    print("Using features for automatic labeling:")
    print(feature_cols)

    logreg, rf = train_models(train_df, feature_cols)
    result_df = auto_label_segments(
        predict_df=predict_df,
        logreg=logreg,
        rf=rf,
        feature_cols=feature_cols,
        probability_threshold=args.threshold,
    )

    print_summary(result_df, top_k=args.top_k)
    save_outputs(result_df, out_dir=args.out_dir, top_k=args.top_k)
    plot_results(result_df, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
