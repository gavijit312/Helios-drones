from typing import List, Tuple
import numpy as np
from src.ingestion.srt_parser import TelemetryFrame


class TelemetryInterpolator:
    def __init__(self, records: List[TelemetryFrame]):
        self.records = records
        self.times = np.array([r.pts_ms for r in records], dtype=np.float64)
        self.lats = np.array([r.latitude for r in records], dtype=np.float64)
        self.lons = np.array([r.longitude for r in records], dtype=np.float64)
        self.alts = np.array([r.altitude_rel for r in records], dtype=np.float64)
        self.pitches = np.array([r.gimbal_pitch for r in records], dtype=np.float64)

    def query_pose(self, query_ms: int) -> Tuple[float, float, float, float]:
        if len(self.records) == 1:
            r = self.records[0]
            return r.latitude, r.longitude, r.altitude_rel, r.gimbal_pitch

        lat = float(np.interp(query_ms, self.times, self.lats))
        lon = float(np.interp(query_ms, self.times, self.lons))
        alt = float(np.interp(query_ms, self.times, self.alts))
        pitch = float(np.interp(query_ms, self.times, self.pitches))
        return lat, lon, alt, pitch