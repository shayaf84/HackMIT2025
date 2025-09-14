import pyaudio
import math
import time
import struct
import json
import cv2
import numpy as np
import multiprocessing

from websockets.sync.server import serve, ServerConnection

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

def sawtooth(t: np.ndarray) -> np.ndarray:
    return (t % 2.0) - 1.0

def sine(t: np.ndarray) -> np.ndarray:
    return np.sin(t * 2 * np.pi)

def smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)

def image_process(x_output: multiprocessing.Queue):
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

        cv2.imshow("frame", frame)

        # Read frames as fast as possible
        cv2.waitKey(1)

def generate_sine_pulse(
    chunk_size: int,
    sample_rate: int,
    sample_offset: int,
    frequency: float
):
    adjusted_volume = max(0.1, min(2.0, math.exp(-(frequency - 440.0) / 880.0)))

    # Send an entire pulse as one chunk
    envelope_samples = 2048
    envelope = np.arange(chunk_size)
    envelope = np.minimum(envelope * 4, chunk_size - 1 - envelope)
    envelope = np.minimum(envelope, envelope_samples)
    envelope = envelope / envelope_samples
    envelope = smoothstep(envelope)

    # Generate pulse
    times = (np.arange(chunk_size) + sample_offset) / sample_rate
    return sine(times * frequency) * adjusted_volume * envelope * 0.2

def audio_process(
    x_input: multiprocessing.Queue,
    server_comms: multiprocessing.Queue,
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

    current_target = None
    success_counter = 0

    pulse_idx = 0

    while True:
        # check for a new target
        first_note = current_target is None

        # wait until the moment we get a target
        # at which we will play that tone
        # then listen for if the player is on that note
        if first_note:
            current_target = server_comms.get()
            pulse_idx = 0
            print("Received current target", current_target)

            sound = generate_sine_pulse(
                chunk_size * 2,
                sample_rate,
                sample_count,
                float(piano_key_frequencies[current_target])
            )
            sample_count += chunk_size * 2

        else:
            # wait until we get data
            while True:
                try:
                    center_x = x_input.get_nowait()
                    xs.append(center_x)
                except:
                    break

            if len(xs) == 0:
                time.sleep(0.01)
                continue

            # grab note from wherever our head is currently pointing
            note_idx: int = math.floor(xs[-1] / FRAME_WIDTH * 48)
            frequency = piano_key_frequencies[note_idx]

            # play probing pulse every 3 times if we are on the wrong note
            if current_target is not None and (pulse_idx % 10 in [0, 3] or current_target == note_idx):
                amplitude_coef = 1.0

                if pulse_idx % 10 == 3:
                    frequency = piano_key_frequencies[current_target]
                    amplitude_coef = 0.8

                sound = generate_sine_pulse(
                    chunk_size,
                    sample_rate,
                    sample_count,
                    float(frequency)
                ) * amplitude_coef
            else:
                sound = np.zeros((chunk_size,))

            sample_count += chunk_size

            pulse_output.put(note_idx)

            if current_target is not None and note_idx == current_target:
                success_counter += 1
                print("success counter:", success_counter)
            else:
                success_counter = 0

        stream.write(struct.pack("%sf" % sound.shape[0], *(sample for sample in sound)))

        # stop playing audio
        if success_counter == 3:
            current_target = None
            pulse_output.put(-1)
            success_counter = 0

        pulse_idx += 1

def start(socket: ServerConnection):
    print("initialized connection!")

    x_queue = multiprocessing.Queue()
    pulse_queue = multiprocessing.Queue()
    server_comms = multiprocessing.Queue()

    img_worker = multiprocessing.Process(
        target=image_process,
        args=(x_queue,)
    )
    img_worker.start()

    audio_worker = multiprocessing.Process(
        target=audio_process,
        args=(x_queue, server_comms, pulse_queue)
    )
    audio_worker.start()

    # listen for start signal
    # client sends "start", with target_idx
    # server sends back data, like which pulses were tried
    # then server sends back "done"!

    while True:
        # recv start signal
        # {
        #   target_note: int,
        # }
        start_signal = json.loads(socket.recv())

        print("Received target note", start_signal["target_note"])

        # send to audio_worker
        server_comms.put(start_signal["target_note"])

        while True:
            pulse_note = pulse_queue.get()
            socket.send(json.dumps({
                "pulse": pulse_note
            }))
            print("Sending pulse", pulse_note)
            if pulse_note == -1:
                break

        # after we are finished, we go back to waiting for start signal

    img_worker.join()

def main():
    with serve(start, host="localhost", port=8000) as server:
        server.serve_forever()

if __name__ == "__main__":
    main()
