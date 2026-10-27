"""Runtime settings loaded from variables.env and environment variables."""
import os
from dataclasses import dataclass
from urllib.parse import quote

from dotenv import load_dotenv


def _env(name, default=None, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return cast(raw.strip())


@dataclass
class Settings:
    # Video source
    source: str
    rtsp_transport: str

    # Detection / tracking
    model_path: str
    target_class: str
    confidence: float
    tracker: str
    device: str

    # Behaviour analysis
    still_threshold: float   # body-sizes per second; below this the animal counts as "still"
    motion_window: float     # seconds of history used to measure movement
    state_hold: float        # seconds a new state must persist before it is logged
    heartbeat_seconds: float  # re-log an unchanged state this often
    track_ttl: float         # forget a track after this many seconds unseen

    # Files
    zones_path: str
    log_path: str

    @classmethod
    def from_env(cls, env_file="variables.env", source=None):
        load_dotenv(env_file)
        return cls(
            source=source or _env("VIDEO_SOURCE") or build_rtsp_url(),
            rtsp_transport=_env("RTSP_TRANSPORT", "tcp"),
            model_path=_env("MODEL", "yolov8n.pt"),
            target_class=_env("TARGET_CLASS", "person"),
            confidence=_env("CONFIDENCE", 0.5, float),
            tracker=_env("TRACKER", "tracker.yaml"),
            device=_env("DEVICE", ""),
            still_threshold=_env("STILL_THRESHOLD", 0.15, float),
            motion_window=_env("MOTION_WINDOW", 2.0, float),
            state_hold=_env("STATE_HOLD", 2.0, float),
            heartbeat_seconds=_env("HEARTBEAT_SECONDS", 600, float),
            track_ttl=_env("TRACK_TTL", 30, float),
            zones_path=_env("ZONES_PATH", "zones.json"),
            log_path=_env("LOG_PATH", "activity_log.csv"),
        )


def build_rtsp_url():
    """Build the camera RTSP URL from environment variables, with URL-encoded credentials."""
    user = _env("CAM_USERNAME")
    password = _env("CAM_PASSWORD") or _env("PASSWORD")
    ip = _env("IP_ADDRESS")
    if not (user and password and ip):
        raise SystemExit(
            "Missing camera settings. Set CAM_USERNAME, PASSWORD and IP_ADDRESS in "
            "variables.env, or set VIDEO_SOURCE to a video file / webcam index."
        )
    port = _env("CAM_PORT", "554")
    stream = _env("CAM_STREAM", "stream1")
    return f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}@{ip}:{port}/{stream}"


def redact(url):
    """Return the URL with its password masked."""
    if "://" in url and "@" in url:
        scheme, rest = url.split("://", 1)
        creds, host = rest.rsplit("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return url


def pick_device(requested):
    """Return the requested device, or the first available of cuda, mps, cpu."""
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
