"""Zone editor: draw polygon zones on a camera frame and save them to zones.json.

    python find_coordinates.py                    # uses sample_frame.jpg
    python find_coordinates.py --image room.jpg
    python find_coordinates.py --grab             # grab a fresh frame from the camera

Controls:
    left click   add a corner and print its coordinates
    right click  undo the last corner
    n            finish the current zone (asks for its name in the terminal)
    d            delete the last saved zone
    s            save to zones.json
    q / Esc      quit
"""
import argparse
import os

import cv2

from zones import Zone, draw_zones, load_zones, save_zones

MAX_WIDTH = 1280
WINDOW = "Zone editor"


def grab_frame():
    from config import Settings
    settings = Settings.from_env()
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{settings.rtsp_transport}")
    cap = cv2.VideoCapture(settings.source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read a frame from the camera")
    cv2.imwrite("sample_frame.jpg", frame)
    print("Saved a fresh sample_frame.jpg")
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default="sample_frame.jpg")
    p.add_argument("--zones", default="zones.json")
    p.add_argument("--grab", action="store_true", help="Grab a new frame from the camera first")
    args = p.parse_args()

    frame = grab_frame() if args.grab else cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Could not open {args.image}")
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_WIDTH / w)  # display scale; clicks are mapped back to frame pixels

    zones = load_zones(args.zones, frame_size=(w, h)) if os.path.exists(args.zones) else []
    current = []

    def on_mouse(event, x, y, flags, param):
        fx, fy = int(x / scale), int(y / scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([fx, fy])
            print(f"Clicked at: ({fx}, {fy})")
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(__doc__)

    while True:
        canvas = draw_zones(frame.copy(), zones)
        for i, pt in enumerate(current):
            cv2.circle(canvas, tuple(pt), 6, (0, 255, 0), -1)
            if i:
                cv2.line(canvas, tuple(current[i - 1]), tuple(pt), (0, 255, 0), 2)
        if scale < 1:
            canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)))
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("n"):
            if len(current) < 3:
                print("A zone needs at least 3 corners")
                continue
            name = input("Zone name (e.g. food zone): ").strip() or f"zone {len(zones) + 1}"
            still = input("Label when the pet is still here (e.g. eating, resting): ").strip() or "resting"
            zones.append(Zone(name, still, current.copy()))
            current.clear()
            print(f"Added '{name}'. Press s to save.")
        elif key == ord("d") and zones:
            print(f"Removed '{zones.pop().name}'")
        elif key == ord("s"):
            save_zones(args.zones, zones, (w, h))
            print(f"Saved {len(zones)} zone(s) to {args.zones}")
        elif key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
