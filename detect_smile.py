#!/usr/bin/env python
import os
import sys

import cv2
import imutils
import numpy as np

if 'linux' in sys.platform:
    import RPi.GPIO as GPIO

    GPIO.setmode(GPIO.BOARD)
    GPIO_PIN = 40
    GPIO.setup(GPIO_PIN, GPIO.OUT)

face_cascade_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
timed_events = [0] * 10


def video_write(filename, frames):
    print(filename, len(frames), frames[0].shape)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the low  er case
    out = cv2.VideoWriter(filename, fourcc, 10.0, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()


class AutoWrite:
    def __init__(self):
        self.files_written = 0
        self.frames_buffer = []
        self.recording_triggered = True
        self.n_frames_in_clip = 160

    def send(self, frame, score):

        # if (not self.recording_triggered) and (score >= 0.3):
        #     self.recording_triggered = True
        #self.recording_triggered = True
        if self.recording_triggered:
            self.frames_buffer.append(frame.copy())
            self.frames_buffer = self.frames_buffer[-self.n_frames_in_clip:]

            if len(self.frames_buffer) == self.n_frames_in_clip:
                folder = 'recording'
                video_write(os.path.join(folder, f'output_{self.files_written:03d}.mp4'), self.frames_buffer)
                self.files_written = (self.files_written + 1) % 10
                self.frames_buffer.clear()
                #self.recording_triggered = False


def add_event(val):
    timed_events.append(val)
    timed_events.pop(0)

auto_writer = AutoWrite()
def detect(frame):
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_detector.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        add_event(0)
    else:
        if len(faces) > 1:
            faces = sorted(faces, key=lambda e: e[3] * e[2], reverse=True)[:1]
        for (x, y, w, h) in faces:
            ax, ay = 0.01, 0.01
            x0_new = int(np.clip(x - ax * w, 0, gray.shape[1] - 1))
            x1_new = int(np.clip(x + w + ax * w, x0_new + 1, gray.shape[1]))
            y0_new = int(np.clip(y - ay * h, 0, gray.shape[0] - 1))
            y1_new = int(np.clip(y + h + ay * h, y0_new + 1, gray.shape[0]))

            # roi_gray = gray[y:y + h, x:x + w]
            roi_gray = gray[y0_new:y1_new, x0_new:x1_new]

            new_aspect_ratio = (x1_new - x0_new) / (y1_new - y0_new)
            golden_width = 192
            golden_brightness = 120
            inferred_height = int(np.round(golden_width / new_aspect_ratio / 4) * 4)

            roi_gray = cv2.resize(roi_gray, (golden_width, inferred_height), interpolation=cv2.INTER_CUBIC)

            roi_gray = np.clip(roi_gray.astype(np.float32) / np.mean(roi_gray) * golden_brightness, 0, 255).astype(
                np.uint8)

            smiles = smile_cascade_detector.detectMultiScale(roi_gray, 1.2, 15)
            if len(smiles) > 0:
                add_event(1)
            else:
                add_event(0)


            cv2.rectangle(frame, [x0_new, y0_new], [x1_new, y1_new], (0, 255, 0), 2)
            score = np.mean(timed_events)
            frame = cv2.putText(frame, f'{score:0.3f}', [x0_new, y0_new], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if 'darwin' in sys.platform:
                cv2.imshow('frame', frame)
                cv2.imshow('roi_gray', roi_gray)

    auto_writer.send(frame, np.mean(timed_events))
    return frame


def control_mirror():
    mean_smile_ind = np.mean(timed_events)
    if 'linux' in sys.platform:
        if mean_smile_ind > 0.3:
            GPIO.output(GPIO_PIN, True)
        else:
            GPIO.output(GPIO_PIN, False)
    else:
        if mean_smile_ind >= 0.3:
            print("Smile detected (%f)" % mean_smile_ind)
        else:
            print("")


def main():
    print("Starting camera")
    #camera = cv2.VideoCapture("/Users/abane7/Downloads/output_000.mp4")
    camera = cv2.VideoCapture(0)

    print("Starting to detect")
    # keep looping
    while True:
        control_mirror()
        # grab the current frame
        (grabbed, frame) = camera.read()
        detect(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # auto_writer = AutoWrite()
        # auto_writer.send(frm, np.mean(timed_events))

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
