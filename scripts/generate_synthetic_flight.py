import cv2
import numpy as np
import math
from pathlib import Path

def generate_sample_flight(
    video_path: str = "data/flight_sample.mp4",
    srt_path: str = "data/flight_sample.srt",
    fps: int = 30,
    duration_sec: int = 8,
    width: int = 640,
    height: int = 480
):
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    total_frames = fps * duration_sec
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    srt_entries = []
    
    base_lat = 52.4862
    base_lon = -1.8904
    base_alt = 45.0
    
    print(f"Synthesizing {duration_sec}s drone flight ({total_frames} frames)...")
    
    for f_idx in range(total_frames):
        t_sec = f_idx / fps
        t_ms = int(t_sec * 1000)
        
        # 1. Synthesize procedural visual frame (aerial flyover landscape)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (45, 95, 40)
        
        # Grid block outlines simulating city roofs
        offset_y = int((f_idx * 6) % 80)
        for y in range(-80 + offset_y, height + 80, 80):
            cv2.line(frame, (0, y), (width, y), (60, 60, 60), 2)
            for x in range(20, width, 100):
                roof_color = (120, 110, 105) if (x + y) % 160 == 0 else (90, 85, 80)
                cv2.rectangle(frame, (x, y + 10), (x + 60, y + 60), roof_color, -1)
                cv2.rectangle(frame, (x, y + 10), (x + 60, y + 60), (30, 30, 30), 2)
        
        noise = np.random.randint(-5, 5, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        writer.write(frame)
        
        # 2. DJI-formatted telemetry record
        lat = base_lat + (f_idx * 0.00002)
        lon = base_lon + (math.sin(f_idx * 0.05) * 0.00001)
        rel_alt = base_alt + math.sin(f_idx * 0.02) * 2.0
        abs_alt = rel_alt + 32.0
        pitch = -25.0 + math.sin(f_idx * 0.05) * 1.5
        yaw = (f_idx * 0.5) % 360.0
        
        ms_start = t_ms
        ms_end = t_ms + int(1000 / fps)
        
        def format_ts(ms):
            h = ms // 3600000
            m = (ms % 3600000) // 60000
            s = (ms % 60000) // 1000
            rem_ms = ms % 1000
            return f"{h:02d}:{m:02d}:{s:02d},{rem_ms:03d}"
        
        entry = (
            f"{f_idx + 1}\n"
            f"{format_ts(ms_start)} --> {format_ts(ms_end)}\n"
            f"[latitude: {lat:.6f}] [longitude: {lon:.6f}] "
            f"[rel_alt: {rel_alt:.2f} abs_alt: {abs_alt:.2f}] "
            f"[pitch: {pitch:.2f}] [yaw: {yaw:.2f}]\n\n"
        )
        srt_entries.append(entry)
        
    writer.release()
    
    with open(srt_path, "w", encoding="utf-8") as f:
        f.writelines(srt_entries)
        
    print(f"Generated sample video: {video_path}")
    print(f"Generated sample telemetry: {srt_path}")

if __name__ == "__main__":
    generate_sample_flight()
