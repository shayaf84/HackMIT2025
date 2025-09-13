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

    rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    height, width, _ = rotated.shape
    relative = rotated / 255.0
    relative -= relative.mean(axis = 2, keepdims = True)
    relative = relative[..., 2]
    relative = np.where(relative > 0.3, relative, 0.0)

    weights = relative.flatten()
    weights_sum = np.sum(weights)


    if weights_sum > 1.0:
        idx = np.arange(height * width)
        xs = idx % width
        ys = idx // width
        center_x = np.sum(xs * weights) / weights_sum
        center_y = np.sum(ys * weights) / weights_sum

        rotated = cv2.ellipse(
            rotated,
            (int(center_x), int(center_y)),
            (10, 10),
            angle = 0,
            startAngle = 0,
            endAngle = 360,
            color = (255, 0, 0),
            thickness = 2
        )
    cv2.imshow("DroidCam Feed", rotated)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

