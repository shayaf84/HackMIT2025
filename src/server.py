import asyncio
import websockets
import cv2
import json
import numpy as np

droidcam_url = "http://<DROIDCAM_IP>/video" 
cap = cv2.VideoCapture(droidcam_url)
if not cap.isOpened():
    print("Error: Could not open DroidCam stream.")
    exit()

async def send_points(websocket):
    print("Connected")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from DroidCam.")

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
                (x,y) = (center_x, center_y)
                data = json.dumps({"x": center_x, "y": center_y})
                await websocket.send(data)
            await asyncio.sleep(0.05)
    except websockets.ConnectionClosed:
        print("disconnected")
    

async def main():
    async with websockets.serve(send_points, "localhost",8000):
        print("WebSocket server running on ws://localhost:8000/")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())