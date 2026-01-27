import cv2 as cv

# Haar cascade — стабильно и без внешних зависимостей
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def face_marker(cap, cam):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80),
        )

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cx = x + w // 2
            cy = y + h // 2
            cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv.imshow("Face mode", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if cam:
            cam.send(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            cam.sleep_until_next_frame()