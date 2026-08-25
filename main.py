import os
# pyrefly: ignore [missing-import]
import cv2
import time

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
    "zones" :
        [{"name": "food zone", "x_min": 20, "y_min":700 , "x_max": 300, "y_max": 1500},
         {"name": "bed zone", "x_min": 560, "y_min":200 , "x_max": 1300, "y_max": 700},
         {"name": "sofa zone", "x_min": 1600, "y_min":200 , "x_max": 1800, "y_max": 1000},
         ]
}

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

    results = model.track(frame, tracker = "bytetrack.yaml", persist = True)

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            label = result.names[int(box.cls)]
            if  label == 'person' and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                if box.id is not None:
                    track_id = int(box.id[0])
                    x = int((x1 + x2) / 2)
                    y = int(y2)
                    lower_box = (x, y)
                    print(f"Box {(x1, y1, x2, y2)} has Track ID: {track_id}")
                    print(f"Lower Box: {lower_box}")

 
                    matched_zone = f"{label.capitalize()}"
                    for zone in zones["zones"]:
                        if lower_box[0] >= zone["x_min"] and lower_box[0] <= zone["x_max"] and lower_box[1] >= zone["y_min"] and lower_box[1] <= zone["y_max"]:
                            matched_zone = f"{label.capitalize()} is in {zone["name"]}"
                            break  

                    color = (0, 0, 255) if matched_zone != label else (0, 0, 255)
                    cv2.putText(frame, matched_zone, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    

    cv2.imshow('Live Dog Detection', frame)
    #click 'q' on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


