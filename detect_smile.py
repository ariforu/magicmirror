#!/usr/bin/env python
import cv2
import imutils
import numpy as np

face_cascade_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
timed_events = [0] * 50

def add_event(val):
    timed_events.append(val)
    timed_events.pop(0)


def detect(gray, frame):
    faces = face_cascade_detector.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:

        roi_gray = gray[y:y + h, x:x + w]
        smiles = smile_cascade_detector.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            add_event(1)
        else:
            add_event(0)
    return frame


def control_mirror():
    print(np.mean(timed_events)>0.3)


def main():
    print("Starting camera")
    camera = cv2.VideoCapture(0)
    print("Starting to detect")
    # keep looping
    while True:
        control_mirror()
        # grab the current frame
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameClone = frame.copy()
        frm = detect(gray, frameClone)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
