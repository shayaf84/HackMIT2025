import cv2

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

    cv2.imshow("DroidCam Feed", rotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

