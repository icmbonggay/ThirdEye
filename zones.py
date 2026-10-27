"""Polygon zones loaded from zones.json.

Points are stored in the pixel coordinates of the reference frame
(frame_width x frame_height) and rescaled to the stream resolution on load.
"""
import json
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Zone:
    name: str
    still_label: str
    points: list  # [[x, y], ...] polygon corners, in order
    color: tuple = (0, 200, 255)
    _contour: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if len(self.points) < 3:
            raise ValueError(f"Zone '{self.name}' needs at least 3 points")
        self._contour = np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)

    def contains(self, point):
        """True if (x, y) is inside or on the edge of the polygon."""
        x, y = point
        return cv2.pointPolygonTest(self._contour, (float(x), float(y)), False) >= 0

    def scaled(self, sx, sy):
        pts = [[round(x * sx), round(y * sy)] for x, y in self.points]
        return Zone(self.name, self.still_label, pts, self.color)


def load_zones(path, frame_size=None):
    """Load zones from JSON, rescaled to frame_size=(width, height) if given."""
    with open(path) as f:
        data = json.load(f)
    zones = [
        Zone(z["name"], z.get("still_label", "resting"), z["points"],
             tuple(z.get("color", (0, 200, 255))))
        for z in data["zones"]
    ]
    if frame_size and data.get("frame_width") and data.get("frame_height"):
        sx = frame_size[0] / data["frame_width"]
        sy = frame_size[1] / data["frame_height"]
        if (sx, sy) != (1, 1):
            zones = [z.scaled(sx, sy) for z in zones]
    return zones


def save_zones(path, zones, frame_size):
    data = {
        "frame_width": frame_size[0],
        "frame_height": frame_size[1],
        "zones": [
            {"name": z.name, "still_label": z.still_label,
             "points": [list(map(int, p)) for p in z.points], "color": list(z.color)}
            for z in zones
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def find_zone(zones, point):
    """First zone containing the point, or None."""
    for zone in zones:
        if zone.contains(point):
            return zone
    return None


def draw_zones(frame, zones, alpha=0.2):
    """Draw zones as labelled, semi-transparent overlays."""
    overlay = frame.copy()
    for z in zones:
        cv2.fillPoly(overlay, [z._contour], z.color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    for z in zones:
        cv2.polylines(frame, [z._contour], True, z.color, 2)
        x, y = z.points[0]
        cv2.putText(frame, z.name, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, z.color, 2)
    return frame
