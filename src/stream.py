import cv2
import numpy as np


# Replace with your DroidCam IP and port
droidcam_url = "http://10.189.79.11:4747/video" 
cap = cv2.VideoCapture(droidcam_url)
if not cap.isOpened():
    print("Error: Could not open DroidCam stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from DroidCam.")
        break

    relative = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    height, width, _ = relative.shape
    relative = relative / 255.0
    relative -= relative.mean(axis = 2, keepdims = True)
    relative = relative[..., 2]
    relative = np.where(relative > 0.4, relative, 0.0)
    relative = np.stack((relative, relative, relative), axis = -1)
    

    cv2.imshow("DroidCam Feed", relative)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

