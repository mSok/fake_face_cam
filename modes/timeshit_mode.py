import cv2 as cv
import numpy as np


def timeshift(cap, cam):
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    COLUMN_WIDTH = 60
    COLUMN_SPEED = 20
    sx = 0
    blank_frame = np.zeros((height, width, 3), np.uint8)  # RGB

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # gray to 3 demension array
        gray = np.repeat(gray[:, :, np.newaxis], 3, axis=2)

        sx = (sx + COLUMN_SPEED) % width
        blank_frame[0:height, sx : COLUMN_WIDTH + sx] = gray[
            0:height, sx : COLUMN_WIDTH + sx
        ]

        # Display the resulting frame
        cv.imshow('copy', blank_frame)

        if cam:
            cam.send(blank_frame)
            cam.sleep_until_next_frame()
        if cv.waitKey(1) == ord('q'):
            break
