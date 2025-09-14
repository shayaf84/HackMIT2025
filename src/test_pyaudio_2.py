# all_frequencies = 2 ** ((np.arange(88) - 49) / 12.0) * 440.0

import pyaudio
import math
import time
import struct
import cv2
import numpy as np
import multiprocessing

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

def sawtooth(t: np.ndarray) -> np.ndarray:
    return (t % 2.0) - 1.0

def sine(t: np.ndarray) -> np.ndarray:
    return np.sin(t * 2 * np.pi)

def smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)

def image_process(
    x_output: multiprocessing.Queue
):
    video_cap = cv2.VideoCapture(0)
    while True:
        ret, frame = video_cap.read()
        height, width, _ = frame.shape

        # Filter out non-red colors
        frame_rel = frame / 255.0
        frame_rel -= frame_rel.mean(axis=2, keepdims=True)
        frame_rel = frame_rel[..., 2]
        frame_rel = np.where(frame_rel > 0.3, frame_rel, 0.0)

        # Compute center of mass of red color
        weights = frame_rel.flatten()
        weights_sum = np.sum(weights)

        # If there is enough mass, send it
        if weights_sum > 1.0:
            idx = np.arange(height * width)
            xs = idx % width
            center_x = np.sum(xs * weights) / weights_sum
            x_output.put(center_x)

        # Read frames as fast as possible
        cv2.waitKey(1)

def audio_process(
    x_input: multiprocessing.Queue,
    pulse_output: multiprocessing.Queue
):
    sample_count = 0
    sample_rate = 48000
    chunk_size = 8192

    piano_key_frequencies = 2 ** ((np.arange(48) - 23) / 12.0) * 440.0

    # keep track of x coordinates of laser
    xs = []

    p = pyaudio.PyAudio()

    stream = p.open(
        format = 1, # 32-bit float
        channels = 1,
        rate = sample_rate,
        output = True
    )

    note_idx = 22

    while True:
        # check for any new xy_input
        while True:
            try:
                center_x = x_input.get_nowait()
                xs.append(center_x)
            except:
                break

        # wait until we get data
        if len(xs) == 0:
            time.sleep(0.1)
            continue

        # grab note from wherever our head is currently pointing
        note_idx: int = math.floor(xs[-1] / FRAME_WIDTH * 48)
        frequency = piano_key_frequencies[note_idx]

        adjusted_volume = max(0.1, min(2.0, math.exp(-(frequency - 440.0) / 880.0)))

        # Send an entire pulse as one chunk
        envelope_samples = 1024
        envelope = np.arange(chunk_size)
        envelope = np.minimum(envelope, chunk_size - 1 - envelope)
        envelope = np.minimum(envelope, envelope_samples)
        envelope = envelope / envelope_samples
        envelope = smoothstep(envelope)

        # Generate pulse
        times = (np.arange(chunk_size) + sample_count) / sample_rate
        sound = sine(times * frequency) * adjusted_volume * envelope * 0.2

        pulse_output.put(note_idx)

        # if note_idx != target_idx:
            # sound *= 0

        stream.write(struct.pack("%sf" % chunk_size, *(sample for sample in sound)))

        sample_count += chunk_size

def main():
    x_queue = multiprocessing.Queue()
    pulse_queue = multiprocessing.Queue()

    img_worker = multiprocessing.Process(target=image_process, args=(x_queue,))
    img_worker.start()

    audio_worker = multiprocessing.Process(target=audio_process, args=(x_queue, pulse_queue))
    audio_worker.start()

    img_worker.join()

if __name__ == "__main__":
    main()
