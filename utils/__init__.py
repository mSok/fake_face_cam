from .face_analyzer import FaceAnalyzer
from .effects import (
    apply_filter,
    adjust_lighting,
    add_overlay,
    color_transfer,
    blend_with_original,
)


def map_value(n, start1, stop1, start2, stop2):
    return (n - start1) / (stop1 - start1) * (stop2 - start2) + start2


__all__ = [
    'FaceAnalyzer',
    'apply_filter',
    'adjust_lighting',
    'add_overlay',
    'color_transfer',
    'blend_with_original',
    'map_value',
]
