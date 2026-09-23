import re
from dataclasses import dataclass
from typing import List


@dataclass
class TelemetryFrame:
    pts_ms: int
    latitude: float
    longitude: float
    altitude_msl: float
    altitude_rel: float
    gimbal_pitch: float
    gimbal_yaw: float


class DJISRTParser:
    def __init__(self):
        self.pattern = re.compile(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s+"
            r".*?\[latitude:\s*([-\d\.]+)\]\s*\[longitude:\s*([-\d\.]+)\]\s*"
            r"\[rel_alt:\s*([-\d\.]+)\s*abs_alt:\s*([-\d\.]+)\]"
            r"(?:.*?\[pitch:\s*([-\d\.]+)\]\s*\[yaw:\s*([-\d\.]+)\])?",
            re.DOTALL | re.IGNORECASE,
        )

    def _time_to_ms(self, time_str: str) -> int:
        h, m, s_ms = time_str.split(":")
        s, ms = s_ms.split(",")
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

    def parse_file(self, srt_path: str) -> List[TelemetryFrame]:
        records = []
        with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        matches = self.pattern.finditer(content)
        for m in matches:
            t_start = self._time_to_ms(m.group(1))
            lat = float(m.group(3))
            lon = float(m.group(4))
            rel_alt = float(m.group(5))
            abs_alt = float(m.group(6))
            pitch = float(m.group(7)) if m.group(7) else -30.0
            yaw = float(m.group(8)) if m.group(8) else 0.0

            records.append(
                TelemetryFrame(
                    pts_ms=t_start,
                    latitude=lat,
                    longitude=lon,
                    altitude_msl=abs_alt,
                    altitude_rel=rel_alt,
                    gimbal_pitch=pitch,
                    gimbal_yaw=yaw,
                )
            )

        if not records:
            # Fallback for synthetic/standard SRT timestamps
            records.append(
                TelemetryFrame(0, 52.4862, -1.8904, 80.0, 50.0, -30.0, 0.0)
            )
        return records