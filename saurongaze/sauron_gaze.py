import logging
import numpy as np
from .ptgaze.common import (Camera, Eye, Face, FaceParts, FacePartsName, MODEL3D,
                     Visualizer)
from .ptgaze.config import get_default_config
from .ptgaze.head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator
from .ptgaze.models import create_model
from .ptgaze.types import GazeEstimationMethod
from .ptgaze.transforms import create_transform
from .ptgaze.gaze_estimator import GazeEstimator
from .ptgaze.utils import update_config, update_default_config, update_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SauronGaze:
    def __init__(self):
        args = { # nothing
            "camera": None,
            "config": None,
            "debug": False,
            "device": None,
            "ext": None,
            "face_detector": 'dlib',
            "image": None,
            "mode": 'eye',
            "no_screen": False,
            "output_dir": None,
            "video": None,
            "face_detector":  None,
        }

        config = get_default_config()
        #print(config)
        update_default_config(config, args)
        config.gaze_estimator.camera_params = "/home/olli/Documents/Personal/pytorch_mpiigaze_demo/saurongaze/ptgaze/data/calib/sample_params.yaml"
        config.gaze_estimator.normalized_camera_params = "/home/olli/Documents/Personal/pytorch_mpiigaze_demo/saurongaze/ptgaze/data/calib/normalized_camera_params_eye.yaml"
        config.face_detector.dlib.model = "/home/olli/.saurongaze/dlib/shape_predictor_68_face_landmarks.dat"
        #print(config)
        
        self._config = config
        self._gaze_estimator = GazeEstimator(config)
        self._visualizer = Visualizer(self._gaze_estimator.camera)
        self._faces = []
        self._gazes = []

    def refresh(self, frame, all_faces=True, copy_frame=True):
        # all_faces=False will only calculate middle face
        # TODO
        # copy_frame=True
        if copy_frame:
            self._visualizer.set_image(frame.copy())
        else:
            self._visualizer.set_image(frame)
        self._faces = self._gaze_estimator.detect_faces(frame)
        for face in self._faces:
            self._gaze_estimator.estimate_gaze(frame, face)

    def draw(self):
        for face in self._faces:
            self._draw_gaze_vector(face)
        return self._visualizer.image

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, faces):
        self._faces = faces

    @property
    def gazes(self):
        return self._gazes

    @gazes.setter
    def gazes(self, gazes):
        self._gazes = gazes

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self._config.demo.gaze_visualization_length
        if self._config.mode == GazeEstimationMethod.MPIIGaze.name:
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self._visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                logger.info(
                    f'[{key.name.lower()}] pitch: {pitch:.2f}, Yaw: {yaw:.2f}')
        elif self._config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            self._visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError