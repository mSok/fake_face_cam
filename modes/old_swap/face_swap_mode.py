import cv2 as cv
import numpy as np
from face_analyze import get_one_face, process_frame, pre_check

def face_swap(cap, cam):
    print('Face swap')
    pre_check()
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    source_img = cv.imread('./source.png')
    source_face = get_one_face(source_img)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        reference_face = get_one_face(frame)

        # cv.rectangle(
        #     frame,
        #     (int(reference_face.bbox[0]), int(reference_face.bbox[1])),
        #     (int(reference_face.bbox[2]), int(reference_face.bbox[3])),
        #     (255,0,0),
        #     2
        # )
        if reference_face:
            frame = process_frame(source_face, reference_face, frame)

        # Display the resulting frame
        cv.imshow('copy', frame)

        if cam:
            cam.send(frame)
            cam.sleep_until_next_frame()


        if cv.waitKey(1) == ord('q'):
            break