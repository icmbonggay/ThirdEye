# ThirdEye

Watches a home camera (Tapo, over RTSP), tracks the pet with YOLOv8 + ByteTrack,
and logs what it is doing and where: *eating in food zone*, *resting in bed zone*, *active*...

## Setup

```bash
python -m venv yolov8 && source yolov8/bin/activate
pip install -r requirements.txt
```

Create `variables.env` (git-ignored):

```
CAM_USERNAME=...
PASSWORD=...
IP_ADDRESS=192.168.x.x
```

## Run

```bash
python main.py                         # live camera with preview (press q to quit)
python main.py --headless              # no window
python main.py --source clip.mp4       # run on a recorded video
python main.py --source clip.mp4 --save-video out.mp4
```

Events go to `activity_log.csv` (timestamp, track_id, activity, zone, speed, reason).
A line is written when the state changes, plus a heartbeat every 10 minutes.

## Zones

Zones live in `zones.json` as polygons, so they can follow the real shape of a bed or sofa.

```bash
python find_coordinates.py --grab      # grab a fresh frame and draw zones on it
```

Click corners, `n` to finish a zone, `s` to save. Zones are rescaled automatically
if the stream resolution changes.

## How "still" vs "active" is decided

* Position = bottom-centre of the box (where the pet touches the floor).
* Speed is measured over a sliding window (`MOTION_WINDOW`, default 2 s).
* Speed is in **body sizes per second** (displacement / √box area).
* A new state must persist for `STATE_HOLD` seconds (default 2 s) before it is logged.

## Tuning (optional, in `variables.env`)

| Variable | Default | Meaning |
|---|---|---|
| `TARGET_CLASS` | `person` | COCO class to track |
| `CONFIDENCE` | `0.5` | minimum detection confidence |
| `STILL_THRESHOLD` | `0.15` | below this many body sizes/s = still (the live label shows the current value) |
| `MOTION_WINDOW` | `2.0` | seconds used to measure speed |
| `STATE_HOLD` | `2.0` | seconds a new state must last before logging |
| `HEARTBEAT_SECONDS` | `600` | re-log an unchanged state this often |
| `DEVICE` | auto | `mps` (Apple GPU), `cuda` or `cpu` |
| `CAM_STREAM` | `stream1` | camera stream path (`stream2` is the low-res stream) |
| `VIDEO_SOURCE` | | use a file / webcam instead of the camera |

## Tests

```bash
pytest
```
