import random
import cv2 as cv
import numpy as np

from utils import map_value

MATRIX_MODE = 2
BR_CHARS = ".-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"


def matrix(cap, cam):
    matrix_mode = MATRIX_MODE

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    chars = BR_CHARS

    cell_width, cell_height = 10, 12
    new_width = int(width / cell_width)
    new_height = int(height / cell_height)
    new_dimension = (new_width, new_height)

    char_offset = new_height - 1
    d_chars = [''.join(random.sample(chars, len(chars))) for _ in range(new_height)]

    while True:
        matrix_window = np.zeros((height, width, 3), dtype=np.uint8)

        ret, frame = cap.read()
        if not ret:
            break

        small_image = cv.resize(frame, new_dimension, interpolation=cv.INTER_NEAREST)
        gray = cv.cvtColor(small_image, cv.COLOR_BGR2GRAY)

        result_image = frame if matrix_mode == 0 else matrix_window

        char_offset -= 1
        for i in range(new_height):
            for j in range(new_width):
                intensive = gray[i][j]
                if intensive < 125:
                    intensive += 40

                match matrix_mode:
                    case 4:
                        char_ind = random.randint(0, len(chars) - 1)
                        color = (0, int(small_image[i][j][1]), 0)
                    case 3:
                        char_ind = j % len(chars)
                        chars = d_chars[(i + char_offset) % len(d_chars)]
                        color = (0, int(small_image[i][j][1]), 0)
                    case 2:
                        char_ind = int(map_value(intensive, 0, 255, 0, len(chars) - 1))
                        color = (0, int(small_image[i][j][1]), 0)
                    case _:
                        char_ind = int(map_value(intensive, 0, 255, 0, len(chars) - 1))
                        color = tuple(map(int, small_image[i][j]))

                cv.putText(
                    img=result_image,
                    text=chars[char_ind],
                    org=(j * cell_width + 15, i * cell_height + 12),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=color,
                    thickness=1,
                )

        cv.imshow('Matrix', result_image)

        key = cv.waitKey(1) & 0xFF
        if key == ord('m'):
            matrix_mode = (matrix_mode + 1) % 5
            print(f'Matrix mode -> {matrix_mode}')
        elif key == ord('q'):
            break

        if cam:
            cam.send(cv.cvtColor(result_image, cv.COLOR_BGR2RGB))
            cam.sleep_until_next_frame()
