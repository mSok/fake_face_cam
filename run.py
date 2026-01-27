import argparse
import time
import cv2 as cv
import pyvirtualcam

from modes.matrix_mode import matrix
from modes.face_mode import face_marker
from modes.raw_mode import raw_camera
from modes.timeshit_mode import timeshift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["matrix", "face", "raw", "timeshift"],
        default="matrix",
        help="Video mode"
    )
    args = parser.parse_args()

    cap = cv.VideoCapture(0)
    time.sleep(1)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = 20

    print(f"Camera: {width}x{height} @ {fps}fps")

    with pyvirtualcam.Camera(
        width=width,
        height=height,
        fps=fps,
        print_fps=True
    ) as cam:

        print(f"VirtualCam: {cam.device}")

        match args.mode:
            case "matrix":
                matrix(cap, cam)
            case "face":
                face_marker(cap, cam)
            case "raw":
                raw_camera(cap, cam)
            case "timeshift":
                timeshift(cap, cam)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()