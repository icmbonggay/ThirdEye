import cv2

frame = cv2.imread("sample_frame.jpg")

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")

cv2.imshow("Click to find coordinates", frame)
cv2.setMouseCallback("Click to find coordinates", on_click)
cv2.waitKey(0)
cv2.destroyAllWindows()