import cv2 as cv


def raw_camera(cap, cam):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow("Raw camera", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if cam:
            cam.send(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            cam.sleep_until_next_frame()