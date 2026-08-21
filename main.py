import os
import cv2
import time
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

    results = model.track(frame, tracker = "bytetrack.yaml")

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            label = result.names[int(box.cls)]
            if  label == 'dog' and conf >= 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Live Dog Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


