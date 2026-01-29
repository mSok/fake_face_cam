import cv2 as cv
import numpy as np
from typing import Tuple, Optional


class EffectType:
    NONE = 'none'
    SHARPEN = 'sharpen'
    CARTOON = 'cartoon'
    VIGNETTE = 'vignette'
    PIXELATE = 'pixelate'
    EDGE_DETECTION = 'edge_detection'
    HUE_SHIFT = 'hue_shift'


def apply_filter(
    frame: np.ndarray, effect_type: str, intensity: float = 1.0
) -> np.ndarray:
    if effect_type == EffectType.NONE:
        return frame

    if effect_type == EffectType.SHARPEN:
        kernel = np.array([[-1, -1, -1], [-1, 9 + intensity, -1], [-1, -1, -1]])
        return cv.filter2D(frame, -1, kernel)

    if effect_type == EffectType.CARTOON:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        edges = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 2
        )
        edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

        color = cv.bilateralFilter(frame, 9, 75, 75)
        return cv.bitwise_and(color, edges)

    if effect_type == EffectType.VIGNETTE:
        return apply_vignette(frame, intensity)

    if effect_type == EffectType.PIXELATE:
        return apply_pixelate(frame, intensity)

    if effect_type == EffectType.EDGE_DETECTION:
        return apply_edge_detection(frame, intensity)

    if effect_type == EffectType.HUE_SHIFT:
        return apply_hue_shift(frame, intensity)

    return frame


def apply_vignette(frame: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    h, w = frame.shape[:2]

    result = frame.copy()

    kernel_size = int(min(h, w) * 0.5 * intensity)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(1, min(100, kernel_size))

    X_result_kernel = cv.getGaussianKernel(w, w // 2)
    Y_result_kernel = cv.getGaussianKernel(h, h // 2)

    kernel = Y_result_kernel * X_result_kernel.T
    mask = kernel / kernel.max()

    mask = cv.resize(mask, (w, h))
    mask = mask[:, :, np.newaxis]

    result = result * mask + result * (1 - mask) * 0.3

    return result.astype(np.uint8)


def apply_pixelate(frame: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    h, w = frame.shape[:2]

    block_size = int(20 * intensity)
    block_size = max(1, min(100, block_size))

    small = cv.resize(
        frame, (w // block_size, h // block_size), interpolation=cv.INTER_NEAREST
    )
    result = cv.resize(small, (w, h), interpolation=cv.INTER_NEAREST)

    return result


def apply_edge_detection(frame: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = edges / edges.max()

    edges = np.uint8(edges * 255)

    threshold = int(50 / intensity)
    threshold = max(10, min(150, threshold))

    _, edges = cv.threshold(edges, threshold, 255, cv.THRESH_BINARY)

    edges_colored = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    return edges_colored


def apply_hue_shift(frame: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    shift = int(intensity * 30)
    shift = shift % 180

    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180

    result = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return result


def adjust_lighting(
    frame: np.ndarray, brightness: float = 0.0, contrast: float = 1.0
) -> np.ndarray:
    adjusted = frame.copy()

    if brightness != 0:
        adjusted = adjusted.astype(np.float32)
        adjusted = adjusted + brightness
        adjusted = np.clip(adjusted, 0, 255)
        adjusted = adjusted.astype(np.uint8)

    if contrast != 1.0:
        adjusted = adjusted.astype(np.float32)
        adjusted = (adjusted - 128) * contrast + 128
        adjusted = np.clip(adjusted, 0, 255)
        adjusted = adjusted.astype(np.uint8)

    return adjusted


def color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    def lab_stats(img):
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        l_mean, l_std = l.mean(), l.std()
        a_mean, a_std = a.mean(), a.std()
        b_mean, b_std = b.mean(), b.std()
        return lab, (l_mean, l_std, a_mean, a_std, b_mean, b_std)

    target_lab, target_stats = lab_stats(target)

    result = source.copy().astype(np.float32)

    for i in range(result.shape[2]):
        result[:, :, i] = (result[:, :, i] - result[:, :, i].mean()) / (
            result[:, :, i].std() + 1e-6
        )

    l_mean, l_std, a_mean, a_std, b_mean, b_std = target_stats

    l, a, b = cv.split(result)
    l = l * (l_std / (l.std() + 1e-6)) + l_mean
    a = a * (a_std / (a.std() + 1e-6)) + a_mean
    b = b * (b_std / (b.std() + 1e-6)) + b_mean

    result_lab = cv.merge([l, a, b])
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

    result_bgr = cv.cvtColor(result_lab, cv.COLOR_LAB2BGR)

    return result_bgr


def blend_with_original(
    original: np.ndarray, swapped: np.ndarray, blend_ratio: float = 0.7
) -> np.ndarray:
    return cv.addWeighted(swapped, blend_ratio, original, 1 - blend_ratio, 0)


def add_overlay(
    frame: np.ndarray,
    overlay_img: Optional[np.ndarray],
    position: Tuple[int, int] = (10, 10),
    size: Tuple[int, int] = (100, 100),
    alpha: float = 0.7,
) -> np.ndarray:
    if overlay_img is None:
        return frame

    overlay = cv.resize(overlay_img, size)
    x, y = position

    if x + size[0] > frame.shape[1] or y + size[1] > frame.shape[0]:
        return frame

    roi = frame[y : y + size[1], x : x + size[0]]

    if overlay.shape[2] == 4:
        overlay_rgb = overlay[:, :, :3]
        overlay_alpha = overlay[:, :, 3] / 255.0
        overlay_alpha = overlay_alpha[:, :, np.newaxis]

        blended = overlay_rgb * overlay_alpha + roi * (1 - overlay_alpha)
        frame[y : y + size[1], x : x + size[0]] = blended.astype(np.uint8)
    else:
        blended = cv.addWeighted(roi, 1 - alpha, overlay, alpha, 0)
        frame[y : y + size[1], x : x + size[0]] = blended

    return frame


def create_face_overlay(face_img: np.ndarray, label: str = '') -> np.ndarray:
    overlay = face_img.copy()

    if label:
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        text_size = cv.getTextSize(label, font, font_scale, thickness)[0]
        bg_height = text_size[1] + 10

        cv.rectangle(overlay, (0, 0), (overlay.shape[1], bg_height), (0, 0, 0), -1)
        cv.putText(
            overlay,
            label,
            (5, text_size[1] + 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    cv.rectangle(
        overlay, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1), (0, 255, 0), 2
    )

    return overlay
