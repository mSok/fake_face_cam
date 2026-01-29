import os
import threading
import urllib.request
from typing import Optional, List, Dict, Any
import numpy as np
import cv2 as cv
import insightface
from insightface.app.common import Face
from tqdm import tqdm


Frame = np.ndarray[Any, Any]
SIMILAR_FACE_DISTANCE = 0.85


class FaceAnalyzer:
    def __init__(self, models_dir: str = './resources/models'):
        self._lock = threading.Lock()
        self._face_analyser = None
        self._face_swapper = None
        self._source_faces: Dict[str, Face] = {}
        self.models_dir = models_dir
        self._ensure_models_dir()

    def _ensure_models_dir(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def _download_model(self, url: str):
        filename = os.path.basename(url)
        filepath = os.path.join(self.models_dir, filename)

        if not os.path.exists(filepath):
            print(f'Downloading {filename}...')
            request = urllib.request.urlopen(url)
            total = int(request.headers.get('Content-Length', 0))

            with tqdm(
                total=total,
                desc='Downloading',
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                urllib.request.urlretrieve(
                    url,
                    filepath,
                    reporthook=lambda count, block_size, total_size: progress.update(
                        block_size
                    ),
                )
            print(f'Downloaded {filename}')

    def _get_face_analyser(self):
        if self._face_analyser is None:
            with self._lock:
                if self._face_analyser is None:
                    self._face_analyser = insightface.app.FaceAnalysis(
                        name='buffalo_l', providers=['CPUExecutionProvider']
                    )
                    self._face_analyser.prepare(ctx_id=0)
        return self._face_analyser

    def _get_face_swapper(self):
        if self._face_swapper is None:
            with self._lock:
                if self._face_swapper is None:
                    model_url = 'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx'
                    self._download_model(model_url)
                    model_path = os.path.join(self.models_dir, 'inswapper_128.onnx')
                    self._face_swapper = insightface.model_zoo.get_model(
                        model_path, providers=['CPUExecutionProvider']
                    )
        return self._face_swapper

    def preload_source_faces(self, face_images: Dict[str, str]):
        for name, image_path in face_images.items():
            if os.path.exists(image_path):
                img = cv.imread(image_path)
                if img is not None:
                    face = self.get_one_face(img)
                    if face:
                        self._source_faces[name] = face
                        print(f'Loaded source face: {name}')
                    else:
                        print(f'Warning: No face detected in {image_path}')

    def get_source_face(self, name: str) -> Optional[Face]:
        return self._source_faces.get(name)

    def get_source_faces_list(self) -> List[str]:
        return list(self._source_faces.keys())

    def get_many_faces(self, frame: Frame) -> Optional[List[Face]]:
        try:
            return self._get_face_analyser().get(frame)
        except Exception:
            return None

    def get_one_face(self, frame: Frame, position: int = 0) -> Optional[Face]:
        many_faces = self.get_many_faces(frame)
        if many_faces:
            try:
                return many_faces[position]
            except IndexError:
                return many_faces[-1]
        return None

    def find_similar_face(self, frame: Frame, reference_face: Face) -> Optional[Face]:
        many_faces = self.get_many_faces(frame)
        if many_faces:
            for face in many_faces:
                if hasattr(face, 'normed_embedding') and hasattr(
                    reference_face, 'normed_embedding'
                ):
                    distance = np.sum(
                        np.square(
                            face.normed_embedding - reference_face.normed_embedding
                        )
                    )
                    if distance < SIMILAR_FACE_DISTANCE:
                        return face
        return None

    def swap_face(self, source_face: Face, target_face: Face, frame: Frame) -> Frame:
        return self._get_face_swapper().get(
            frame, target_face, source_face, paste_back=True
        )

    def swap_faces_in_frame(
        self,
        frame: Frame,
        source_face_name: str,
        skip_frames: int = 0,
        frame_count: int = 0,
    ) -> Frame:
        source_face = self.get_source_face(source_face_name)
        if not source_face:
            return frame

        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            return None

        target_face = self.get_one_face(frame)
        if target_face:
            return self.swap_face(source_face, target_face, frame)

        return frame
