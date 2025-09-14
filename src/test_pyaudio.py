from multiprocessing import connection
import pyaudio
import math
import struct
import cv2
import numpy as np
import multiprocessing

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

def image_process(xy_output: multiprocessing.Queue):
    video_cap = cv2.VideoCapture(0)
    while True:
        ret, frame = video_cap.read()
        height, width, _ = frame.shape

        # Detect x, y coordinates of laser
        frame_rel = frame / 255.0
        frame_rel -= frame_rel.mean(axis=2, keepdims=True)
        frame_rel = frame_rel[..., 2]
        frame_rel = np.where(frame_rel > 0.3, frame_rel, 0.0)

        weights = frame_rel.flatten()
        weights_sum = np.sum(weights)

        if weights_sum > 1.0:
            idx = np.arange(height * width)
            xs = idx % width
            ys = idx // width
            center_x = np.sum(xs * weights) / weights_sum
            center_y = np.sum(ys * weights) / weights_sum
            print("Found", center_x, center_y)
            xy_output.put((int(center_x), int(center_y)))

        cv2.waitKey(1)

def audio_process(xy_input: multiprocessing.Queue):

    sample_count = 0
    sample_rate = 48000
    chunk_size = 512

    bucket_size_pixels = 5
    num_buckets_x = FRAME_WIDTH // bucket_size_pixels
    num_buckets_y = FRAME_HEIGHT // bucket_size_pixels

    smooth_x = FRAME_WIDTH * 0.5
    smooth_y = FRAME_HEIGHT * 0.5

    last_seen_x = FRAME_WIDTH * 0.5
    last_seen_y = FRAME_HEIGHT * 0.5

    # manage weights
    x_bucket_weights = np.zeros((num_buckets_x,))
    y_bucket_weights = np.zeros((num_buckets_y,))

    #x_bucket_weights[0] = 1.0
    #y_bucket_weights[0] = 1.0

    target_x_bucket_weights = x_bucket_weights * 0.0
    target_y_bucket_weights = y_bucket_weights * 0.0

    p = pyaudio.PyAudio()

    stream = p.open(
        format = 1, # 32-bit float
        channels = 1,
        rate = sample_rate,
        output = True
    )

    def sawtooth(t: np.ndarray) -> np.ndarray:
        return (t % 2.0) - 1.0

    def sine(t: np.ndarray) -> np.ndarray:
        return np.sin(t * 2 * np.pi)

    frequencies = np.array([
        440.0, # A4
        659.2551, # E5
        523.2511, # C5
        783.9909, # G5
    ])[None, :, None]

    while True:
        # check for any new xy_input
        while True:
            try:
                center_x, center_y = xy_input.get_nowait()

                last_seen_x = center_x
                last_seen_y = center_y

                x_bucket = int(center_x / bucket_size_pixels)
                y_bucket = int(center_y / bucket_size_pixels)

                target_x_bucket_weights *= 0
                target_y_bucket_weights *= 0

                target_x_bucket_weights[x_bucket] = 1
                target_y_bucket_weights[y_bucket] = 1

            except:
                break

        weights = np.array([
            smooth_x / FRAME_WIDTH,
            1.0 - smooth_x / FRAME_WIDTH,
            smooth_y / FRAME_HEIGHT,
            1.0 - smooth_y / FRAME_HEIGHT
        ])[None, :]

        num_harmonics = 1.0

        harmonics = 2**np.arange(num_harmonics)[None, None, :]
        harmonic_weights = (num_harmonics - np.arange(num_harmonics, dtype=np.float64))[None, None, :]
        harmonic_weights /= harmonic_weights.sum()

        times = (np.arange(chunk_size) + sample_count)[:, None, None] / sample_rate

        sound = np.sum(np.sum(sine(times * frequencies * harmonics) * harmonic_weights, axis=-1) * weights, axis=-1)

        """
        x_bucket_weights += (target_x_bucket_weights - x_bucket_weights) * 0.4
        y_bucket_weights += (target_y_bucket_weights - y_bucket_weights) * 0.4

        times = (np.arange(chunk_size) + sample_count)[:, None, None] / sample_rate

        num_harmonics = 3

        x_frequencies = (np.arange(num_buckets_x) * bucket_size_pixels * 0.2 + 44)[None, :, None] * (np.arange(num_harmonics))[None, None, :]
        y_frequencies = (np.arange(num_buckets_y) * bucket_size_pixels * 0.2 + 350)[None, :, None] * (np.arange(num_harmonics))[None, None, :]

        harmonic_weights = (num_harmonics * 2.0 - np.arange(num_harmonics, dtype=np.float64))[None, None, :]
        harmonic_weights /= harmonic_weights.sum()

        x_sound = ((sawtooth(times * x_frequencies) * harmonic_weights).sum(axis=2) * x_bucket_weights * 0.1).sum(axis=1)
        y_sound = ((sawtooth(times * y_frequencies) * harmonic_weights).sum(axis=2) * y_bucket_weights * 0.05).sum(axis=1)

        sound: np.ndarray = x_sound + y_sound
        """

        # generate a sound for each corner, modulate through distances


        smooth_x += (last_seen_x - smooth_x) * 0.45
        smooth_y += (last_seen_y - smooth_y) * 0.45

        stream.write(struct.pack("%sf" % chunk_size, *(sample for sample in sound)))
        sample_count += chunk_size



def main():
    queue = multiprocessing.Queue()
    img_worker = multiprocessing.Process(target=image_process, args=(queue,))
    img_worker.start()

    audio_worker = multiprocessing.Process(target=audio_process, args=(queue,))
    audio_worker.start()

    img_worker.join()

def main_old_2():
    sample_count = 0
    sample_rate = 48000
    frequency = 150
    chunk_size = 1024

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
        for i in range(1):
            new_freq = frequency + i * 0.1
            if i == 1:
                new_freq = 200
            t_rel = idx / sample_rate * new_freq
            combined += ((t_rel % 2.0) - 1.0) * 0.1
        return combined / 10

    while True:
        frequency += 2
        buf = struct.pack('%sf' % chunk_size, *(
            sample_to_value_3(sample_count + i) for i in range(chunk_size)
        ))
        stream.write(buf)
        print(frequency)
        sample_count += chunk_size
    pass

def main_old():
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
