import logging
from pathlib import Path
import numpy as np
from torch.cuda import is_available as cuda_available
from types import SimpleNamespace
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


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)

    def __repr__(self):
        items = []
        for k, v in self.__dict__.items():
            if isinstance(v, NestedNamespace):
                items.append(f"{k}: {{\n" + "\n  ".join(("  " + v.__repr__()).split('\n')) + f"\n}},")
            else:
                items.append(f"{k}: {v!r},")
        return "{}".format("\n".join(items))


class SauronGaze:
    def __init__(self, config=None): 
        # TODO utils.py download scripts
        self._config = config if config else self.get_default_config()
        self._gaze_estimator = GazeEstimator(self._config)
        self._visualizer = Visualizer(self._gaze_estimator.camera)
        self._faces = []

    @staticmethod
    def get_default_config(cuda=False):
        if cuda and not cuda_available():
            raise RuntimeError("CUDA not available")

        saurongaze_path = Path(__file__).resolve().parent

        return NestedNamespace({
            "demo": {
                "display_on_screen": True,
                "gaze_visualization_length": 0.05,
                "head_pose_axis_length": 0.05,
                "image_path": None,
                "output_dir": None,
                "output_file_extension": "avi",
                "show_bbox": True,
                "show_head_pose": True,
                "show_landmarks": False,
                "show_normalized_image": False,
                "show_template_model": False,
                "use_camera": True,
                "video_path": None,
                "wait_time": 1,
            },
            "device": "cuda" if cuda else "cpu",
            "face_detector": {
                "dlib": {
                    "model": "/home/olli/.saurongaze/dlib/shape_predictor_68_face_landmarks.dat" # TODO different way
                }, 
                "mode": "dlib",
            },
            "gaze_estimator": {
                "camera_params": f"{saurongaze_path}/ptgaze/data/calib/sample_params.yaml",
                "checkpoint": f"/home/olli/.saurongaze/models/mpiigaze_resnet_preact.pth", # TODO different way
                "normalized_camera_distance": 0.6,
                "normalized_camera_params": f"{saurongaze_path}/ptgaze/data/calib/normalized_camera_params_eye.yaml",
            },
            "mode": "MPIIGaze",
            "model": {
                "backbone": {
                    "name": "resnet_simple",
                    "pretrained": "resnet18",
                    "resnet_block": "basic",
                    "resnet_layers": [2, 2, 2],
                },
                "name": "resnet_preact",
            },
            "transform": {
                "mpiifacegaze_face_size": 224,
                "mpiifacegaze_gray": False,
            },
        })

    def refresh(self, frame, copy_frame=False):
        if copy_frame:
            self._visualizer.set_image(frame.copy())
        else:
            self._visualizer.set_image(frame)
        self._faces = self._gaze_estimator.detect_faces(frame)

        if len(self._faces) > 0:
            # TODO select the middle face only
            self._gaze_estimator.estimate_gaze(frame, self._faces[0])
            
            euler_angles = self._faces[0].head_pose_rot.as_euler('XYZ', degrees=True)
            head_pitch, head_yaw, head_roll = self._faces[0].change_coordinate_system(euler_angles)
            head = NestedNamespace({
                "pitch": head_pitch,
                "yaw": head_yaw,
                "roll": head_roll,
                "distance": self._faces[0].distance,
            })

            right_eye = getattr(self._faces[0], FacePartsName.REYE.name.lower())
            left_eye = getattr(self._faces[0], FacePartsName.LEYE.name.lower())
            right_eye_pitch, right_eye_yaw = np.rad2deg(right_eye.vector_to_angle(right_eye.gaze_vector))
            left_eye_pitch, left_eye_yaw = np.rad2deg(left_eye.vector_to_angle(left_eye.gaze_vector))
            gaze = NestedNamespace({
                "right": {
                    "pitch": right_eye_pitch,
                    "yaw": right_eye_yaw,
                },
                "left": {
                    "pitch": left_eye_pitch,
                    "yaw": left_eye_yaw,
                },
            })

            return head, gaze
        else:
            return None, None

    def get_frame(self, head=True, gaze=True):
        if len(self._faces) > 0:
            if head:
                self._draw_head_pose(self._faces[0])
            if gaze:
                self._draw_gaze_vector(self._faces[0])
        return self._visualizer.image

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self._config.demo.gaze_visualization_length
        if self._config.mode == GazeEstimationMethod.MPIIGaze.name:
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self._visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
        elif self._config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            raise RuntimeError("MPIIFaceGaze not supported")
            """
            self._visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
            """
        else:
            raise ValueError

    def _draw_head_pose(self, face: Face) -> None:
        if not self._config.demo.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self._config.demo.head_pose_axis_length
        self._visualizer.draw_model_axes(face, length, lw=2)
