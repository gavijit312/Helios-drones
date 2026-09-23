import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def prepare_zurich_dataset(
    images_dir: str = "data/zurich",
    gps_csv: str = "data/zurich/Log Files/OnbordGPS.csv",
    output_mp4: str = "data/zurich_flight.mp4",
    output_srt: str = "data/zurich_flight.srt",
    max_frames: int = 150,
    fps: float = 15.0,
):
    in_path = Path(images_dir)
    Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)
# Search recursively for image frames (.jpg, .png, .jpeg)
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    img_files = []
    for ext in extensions:
        img_files.extend(list(in_path.rglob(ext)))

    # Ignore macOS metadata files (starting with '._' or '.')
    img_files = [p for p in img_files if not p.name.startswith("._") and not p.name.startswith(".")]

    # Sort numerically/alphabetically by filename
    img_files = sorted(img_files, key=lambda p: p.name)

    if len(img_files) == 0:
        print(f"[Error] No image frames found under {images_dir}.")
        print("Please extract the Zurich dataset images into that directory.")
        return

    selected_files = img_files[:max_frames]
    print(f"[Zurich Ingest] Found {len(img_files)} total frames. Processing {len(selected_files)} frames.")

    first_img = cv2.imread(str(selected_files[0]))
    if first_img is None:
        print(f"[Error] Failed to read first image: {selected_files[0]}")
        return
    h, w, _ = first_img.shape

    # 1. Compile video sequence
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_mp4, fourcc, fps, (w, h))

    print(f"[Zurich Ingest] Compiling frames into {output_mp4} ({w}x{h} @ {fps} FPS)...")
    for fpath in selected_files:
        frame = cv2.imread(str(fpath))
        if frame is not None:
            writer.write(frame)
    writer.release()
    print("[Zurich Ingest] Video compilation complete.")

    # 2. Build synchronized telemetry
    gps_path = Path(gps_csv)
    df_gps = None
    if gps_path.exists():
        try:
            df_gps = pd.read_csv(gps_path)
            print(f"[Zurich Ingest] Loaded GPS records from {gps_path}")
        except Exception as e:
            print(f"[Zurich Ingest] Warning: Could not read CSV ({e}). Using synthetic trajectory.")

    def fmt_time(ms):
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        rem_ms = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{rem_ms:03d}"

    interval_ms = int(1000 / fps)
    total_duration_ms = int(len(selected_files) * interval_ms)
    srt_entries = []

    # Zurich coordinates reference (Old Town Zurich)
    base_lat = 47.3769
    base_lon = 8.5417
    base_alt = 12.0  # Urban MAV flies 8m-15m above ground

    for idx in range(len(selected_files)):
        t_start = idx * interval_ms
        t_end = min((idx + 1) * interval_ms, total_duration_ms)

        if df_gps is not None and idx < len(df_gps):
            lat = float(df_gps.iloc[idx].get("Latitude", base_lat))
            lon = float(df_gps.iloc[idx].get("Longitude", base_lon))
            alt = float(df_gps.iloc[idx].get("Altitude", base_alt))
        else:
            lat = base_lat + (idx * 0.000008)
            lon = base_lon + (idx * 0.000010)
            alt = base_alt + (np.sin(idx * 0.1) * 0.5)

        # Zurich MAV camera is mounted forward-facing horizontal (pitch near 0 deg)
        pitch = 0.0
        yaw = 45.0

        entry = (
            f"{idx + 1}\n"
            f"{fmt_time(t_start)} --> {fmt_time(t_end)}\n"
            f"[latitude: {lat:.6f}] [longitude: {lon:.6f}] "
            f"[rel_alt: {alt:.2f} abs_alt: {alt + 408.0:.2f}] "
            f"[pitch: {pitch:.2f}] [yaw: {yaw:.2f}]\n\n"
        )
        srt_entries.append(entry)

    with open(output_srt, "w", encoding="utf-8") as f:
        f.writelines(srt_entries)

    print(f"[Zurich Ingest] Telemetry written to {output_srt}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="data/zurich")
    parser.add_argument("--gps_csv", default="data/zurich/Log Files/OnbordGPS.csv")
    parser.add_argument("--out_video", default="data/zurich_flight.mp4")
    parser.add_argument("--out_srt", default="data/zurich_flight.srt")
    parser.add_argument("--frames", type=int, default=150)
    args = parser.parse_args()

    prepare_zurich_dataset(
        images_dir=args.images_dir,
        gps_csv=args.gps_csv,
        output_mp4=args.out_video,
        output_srt=args.out_srt,
        max_frames=args.frames,
    )