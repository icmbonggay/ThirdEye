"""Third-Eye: watch a camera, track the pet, and log what it's doing and where.

Usage:
    python main.py                      # camera from variables.env, with a preview window
    python main.py --headless           # no window (e.g. running on a server)
    python main.py --source clip.mp4    # test on a recorded video
    python main.py --source 0           # test on the laptop webcam
"""
import argparse
import csv
import logging
import math
import os
import time

import cv2
from ultralytics import YOLO

from activity import MotionHistory, StateDebouncer, TrackState, classify, describe
from config import Settings, pick_device, redact
from stream import FrameReader
from zones import draw_zones, find_zone, load_zones

log = logging.getLogger("third-eye")

BOX_COLOR = (0, 0, 255)
MAX_PREVIEW_WIDTH = 1280
WINDOW_NAME = "Live Dog Detection"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="Video file, webcam index or RTSP URL (overrides variables.env)")
    p.add_argument("--env", default="variables.env", help="Settings file (default: variables.env)")
    p.add_argument("--headless", action="store_true", help="Don't open a preview window")
    p.add_argument("--save-video", metavar="PATH", help="Write the annotated video to this file")
    return p.parse_args()


class ActivityLog:
    """Append-only CSV activity log."""

    FIELDS = ["timestamp", "track_id", "label", "activity", "zone", "speed", "reason", "description"]

    def __init__(self, path):
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        self._file = open(path, "a", newline="")
        self._writer = csv.writer(self._file)
        if new_file:
            self._writer.writerow(self.FIELDS)
            self._file.flush()

    def write(self, ts, track_id, label, state, speed, reason):
        activity, zone_name = state
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        self._writer.writerow([stamp, track_id, label, activity, zone_name or "",
                               f"{speed:.3f}", reason, describe(label, state)])
        self._file.flush()
        log.info("%s  #%d  %s (%s)", stamp, track_id, describe(label, state), reason)

    def close(self):
        self._file.close()


def resolve_class_id(model, name):
    for class_id, class_name in model.names.items():
        if class_name == name:
            return class_id
    raise SystemExit(f"'{name}' is not a class this model knows. Options: {sorted(model.names.values())}")


def annotate(frame, box, track_id, text):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    cv2.circle(frame, ((x1 + x2) // 2, y2), 5, BOX_COLOR, -1)  # zone reference point
    label = f"#{track_id} {text}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    top = max(y1 - h - 10, 0)
    cv2.rectangle(frame, (x1, top), (x1 + w + 6, top + h + 10), BOX_COLOR, -1)
    cv2.putText(frame, label, (x1 + 3, top + h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    settings = Settings.from_env(args.env, source=args.source)

    device = pick_device(settings.device)
    model = YOLO(settings.model_path)
    class_id = resolve_class_id(model, settings.target_class)
    label = settings.target_class
    log.info("Source: %s | model: %s | device: %s | tracking: %s",
             redact(settings.source), settings.model_path, device, label)

    reader = FrameReader(settings.source, rtsp_transport=settings.rtsp_transport).start()
    activity_log = ActivityLog(settings.log_path)
    tracks = {}
    zones = None
    writer = None
    fps_avg = None
    last_tick = time.time()

    try:
        while True:
            frame, ts = reader.read()
            if frame is None:
                break

            if zones is None:  # loaded on the first frame to match its resolution
                h, w = frame.shape[:2]
                if os.path.exists(settings.zones_path):
                    zones = load_zones(settings.zones_path, frame_size=(w, h))
                else:
                    log.warning("%s not found, running without zones (create it with find_coordinates.py)",
                                settings.zones_path)
                    zones = []

            result = model.track(frame, persist=True, tracker=settings.tracker, classes=[class_id],
                                 conf=settings.confidence, device=device, verbose=False)[0]

            draw_zones(frame, zones)
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for xyxy, track_id in zip(boxes.xyxy.int().tolist(), boxes.id.int().tolist()):
                    x1, y1, x2, y2 = xyxy
                    foot = ((x1 + x2) // 2, y2)  # bottom-centre of the box
                    scale = math.sqrt(max((x2 - x1) * (y2 - y1), 1))

                    track = tracks.get(track_id)
                    if track is None:
                        track = tracks[track_id] = TrackState(
                            MotionHistory(settings.motion_window), StateDebouncer(settings.state_hold))
                    track.history.add(ts, foot[0], foot[1], scale)

                    speed = track.history.speed()
                    if speed is None:
                        annotate(frame, xyxy, track_id, "watching...")
                        continue

                    zone = find_zone(zones, foot)
                    changed = track.debouncer.update(classify(speed, zone, settings.still_threshold), ts)
                    state = track.debouncer.state
                    if changed:
                        activity_log.write(ts, track_id, label, state, speed, "change")
                        track.last_logged = ts
                    elif ts - track.last_logged >= settings.heartbeat_seconds:
                        activity_log.write(ts, track_id, label, state, speed, "heartbeat")
                        track.last_logged = ts

                    annotate(frame, xyxy, track_id, f"{describe(label, state)} ({speed:.2f})")

            # Remove tracks not seen within TRACK_TTL seconds.
            for tid in [t for t, s in tracks.items() if ts - s.history.last_seen > settings.track_ttl]:
                del tracks[tid]

            now = time.time()
            inst = 1.0 / max(now - last_tick, 1e-6)
            fps_avg = inst if fps_avg is None else 0.9 * fps_avg + 0.1 * inst
            last_tick = now
            cv2.putText(frame, f"{fps_avg:.1f} FPS", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if args.save_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             reader.fps or 15, (w, h))
                writer.write(frame)

            if not args.headless:
                h, w = frame.shape[:2]
                if w > MAX_PREVIEW_WIDTH:
                    frame = cv2.resize(frame, (MAX_PREVIEW_WIDTH, int(h * MAX_PREVIEW_WIDTH / w)))
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # press 'q' to quit
                    break
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        activity_log.close()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    if reader.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
