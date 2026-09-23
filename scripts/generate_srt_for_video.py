from pathlib import Path
import cv2

def create_telemetry(video_path="data/real_testing_drone.mp4", srt_out="data/real_testing_drone.srt"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_ms = int((total_frames / fps) * 1000)
    print(f"[Telemetry] Matched to video: {total_frames} frames ({duration_ms / 1000:.1f}s @ {fps:.1f} FPS)")

    def fmt_time(ms):
        h = ms // 3600000
        m = (ms % 3600000) // 60000
        s = (ms % 60000) // 1000
        rem = ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d},{rem:03d}"

    entries = []
    interval = int(1000 / fps)
    
    # Starting coordinates and flight parameters
    lat, lon, alt = 22.6530, 88.4230, 45.0

    for i, t in enumerate(range(0, duration_ms, interval)):
        t_end = min(t + interval, duration_ms)
        cur_lat = lat + (i * 0.00001)
        cur_lon = lon + (i * 0.000008)
        cur_alt = alt + (i * 0.004)
        
        entry = (
            f"{i + 1}\n"
            f"{fmt_time(t)} --> {fmt_time(t_end)}\n"
            f"[latitude: {cur_lat:.6f}] [longitude: {cur_lon:.6f}] "
            f"[rel_alt: {cur_alt:.2f} abs_alt: {cur_alt + 15.0:.2f}] "
            f"[pitch: -30.00] [yaw: {(i * 0.2) % 360:.2f}]\n\n"
        )
        entries.append(entry)

    with open(srt_out, "w", encoding="utf-8") as f:
        f.writelines(entries)
    print(f"[Telemetry] Saved: {srt_out}")

if __name__ == "__main__":
    create_telemetry()
