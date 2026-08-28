import os
# pyrefly: ignore [missing-import]
import cv2
import time
import math
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv("variables.env")

model = YOLO('yolov8n.pt')
USERNAME = os.environ.get("CAM_USERNAME")
PASSWORD = os.environ.get("PASSWORD")
IP_ADDRESS = os.environ.get("IP_ADDRESS")
rtsp = f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}:554/stream1"

cap = cv2.VideoCapture(rtsp)
counter = 0

zones = {
    "zones": [
        {"name": "food zone", "x_min": 20, "y_min": 700, "x_max": 300, "y_max": 1500, "still_label": "eating"},
        {"name": "bed zone", "x_min": 560, "y_min": 200, "x_max": 1300, "y_max": 700, "still_label": "resting"},
        {"name": "sofa zone", "x_min": 1600, "y_min": 200, "x_max": 1800, "y_max": 1000, "still_label": "resting"},
    ]
}

previous_positions = {}
last_logged_state = {} 
last_log_time = {}       
HEARTBEAT_SECONDS = 600  

while True:
    success, frame = cap.read()
    if not success:
        while counter < 4:
            cap = cv2.VideoCapture(rtsp)
            success, frame = cap.read()
            if success:
                counter = 0
                break
            counter += 1
            time.sleep(30)
        else:
            print("Lost connection to CCTV. Program will exit now...")
            break

    results = model.track(frame, tracker="bytetrack.yaml", persist=True)

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            label = result.names[int(box.cls)]
            if label == 'person' and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                if box.id is not None:
                    track_id = int(box.id[0])
                    x = int((x1 + x2) / 2)
                    y = int(y2)
                    lower_box = (x, y)
                    print(f"Box {(x1, y1, x2, y2)} has Track ID: {track_id}")
                    print(f"Lower Box: {lower_box}")

                    last_seen = previous_positions.get(track_id)
                    previous_positions.update({track_id: (x, y, time.time())})

                    if last_seen is not None:
                        prev_x, prev_y, prev_time = last_seen
                        distance = math.dist((prev_x, prev_y), (x, y))
                        elapsed = time.time() - prev_time
                        speed = distance / elapsed if elapsed > 0 else 0
                    else:
                        speed = 0

                    print(f"Speed: {speed:.1f} px/sec")

                    mode = "resting" if speed < 20 else "active"

                    matched_zone = f"{label.capitalize()} is {mode}"
                    for zone in zones["zones"]:
                        if (lower_box[0] >= zone["x_min"] and lower_box[0] <= zone["x_max"] and
                                lower_box[1] >= zone["y_min"] and lower_box[1] <= zone["y_max"]):
                            zone_mode = zone["still_label"] if speed < 20 else "active"
                            matched_zone = f"{label.capitalize()} is {zone_mode} in {zone['name']}"
                            break

                    now = time.time()
                    state_changed = last_logged_state.get(track_id) != matched_zone
                    heartbeat_due = now - last_log_time.get(track_id, 0) >= HEARTBEAT_SECONDS

                    if state_changed or heartbeat_due:
                        with open("log.txt", "a") as file:
                            file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {matched_zone}\n")
                        last_logged_state[track_id] = matched_zone
                        last_log_time[track_id] = now

                    cv2.putText(frame, matched_zone, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Live Dog Detection', frame)
    # click 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()