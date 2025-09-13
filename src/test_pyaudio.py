import pyaudio
import math
import struct
import cv2
import numpy as np

def main_old():
    sample_count = 0
    sample_rate = 48000
    frequency = 150
    chunk_size = 2048

    p = pyaudio.PyAudio()

    stream = p.open(
        format = 1, # 32-bit float
        channels = 1,
        rate = sample_rate,
        output = True
    )

    def sample_to_value(idx: int):
        t_seconds = idx / sample_rate
        return math.sin( t_seconds * frequency * math.pi * 2 ) * 0.2

    def sample_to_value_2(idx: int):
        t_rel = idx / sample_rate * frequency
        return ((t_rel % 2.0) - 1.0) * 0.1

    def sample_to_value_3(idx: int):
        combined = 0
        for i in range(10):
            new_freq = frequency + i * 0.1
            if i == 1:
                new_freq = 200
            t_rel = idx / sample_rate * new_freq
            combined += ((t_rel % 2.0) - 1.0) * 0.1
        return combined / 10

    for i in range(300):
        # frequency += 2

        samples = [ sample_to_value_3(sample_count + i) for i in range(chunk_size) ]
        buf = struct.pack('%sf' % chunk_size, *samples)
        stream.write(buf)
        sample_count += chunk_size

    pass

def main():
    video_cap = cv2.VideoCapture(0)

    while True:
        ret, frame = video_cap.read()
        height, width, _ = frame.shape
        frame_rel = frame / 255.0
        frame_rel -= frame_rel.mean(axis=2, keepdims=True)
        frame_rel = frame_rel[..., 2]
        frame_rel = np.where(frame_rel > 0.4, frame_rel, 0.0)

        weights = frame_rel.flatten()
        weights_sum = np.sum(weights)

        if weights_sum > 1.0:
            idx = np.arange(height * width)
            xs = idx % width
            ys = idx // width
            center_x = np.sum(xs * weights) / weights_sum
            center_y = np.sum(ys * weights) / weights_sum

            frame = cv2.ellipse(
                frame,
                (int(center_x), int(center_y)),
                (10, 10),
                angle = 0,
                startAngle = 0,
                endAngle = 360,
                color = (255, 0, 0),
                thickness = 2
            )

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    main()
