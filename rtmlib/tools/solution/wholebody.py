import time
from typing import List, Optional
import cv2
import numpy as np

from .. import YOLOX, RTMPose, RTMDet, RTMDetRegional
from .utils.types import BodyResult, Keypoint, PoseResult


class Wholebody:

    MODE = {
        'performance': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip',  # noqa
            'pose_input_size': (288, 384),
        },
        'lightweight': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip',  # noqa
            'det_input_size': (416, 416),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip',  # noqa
            'pose_input_size': (192, 256),
        },
        'balanced': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-256x192-fbef0d61_20230925.zip',  # noqa
            'pose_input_size': (192, 256),
        },
        'lightweight_rtm': {
            'det':
            '/home/y0f01wf/lazy_susan/lazy_susan_inference/rtmpose-ort/rtmdet-nano/end2end.onnx',  # noqa
            'det_input_size': (320, 320),
            'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.zip',
            # "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip",
            # 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.zip',
            # 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.zip',  # noqa
            'pose_input_size': (192, 256),
        },
    }

    def __init__(self,
                 det: str = None,
                 det_input_size: tuple = (640, 640),
                 pose: str = None,
                 pose_input_size: tuple = (288, 384),
                 mode: str = 'balanced',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        
        self.mode = mode

        if det is None:
            det = self.MODE[mode]['det']
            det_input_size = self.MODE[mode]['det_input_size']

        if pose is None:
            pose = self.MODE[mode]['pose']
            pose_input_size = self.MODE[mode]['pose_input_size']

        if 'rtm' in mode:
            self.do_flip = True
            self.det_model = RTMDet(det,
                                model_input_size=det_input_size,
                                backend=backend,
                                device=device)
        else:
            self.do_flip = False
            self.det_model = YOLOX(det,
                                model_input_size=det_input_size,
                                backend=backend,
                                device=device)
        
        self.pose_model = RTMPose(pose,
                                model_input_size=pose_input_size,
                                to_openpose=to_openpose,
                                backend=backend,
                                device=device)

    def __call__(self, image: np.ndarray):
        """One inference for upper image (with some buffer). One for lower.
        WARNING: there is no dedup here.
        """
        if not self.do_flip:
            bboxes = self.det_model(image)
            keypoints, scores = self.pose_model(image, bboxes=bboxes)
        else:
            img_h, img_w, _ =  image.shape
            upper_image = np.copy(image)
            upper_image[int(img_h / 2 * 1.2):, :] = 255.
            lower_image = cv2.flip(image, 0)
            lower_image[int(img_h / 2 * 1.2):, :] = 255.

            start_time = time.time()
            upper_bboxes = self.det_model(upper_image)
            lower_bboxes = self.det_model(lower_image)
            print(f"det_time:{time.time() - start_time}s")
            start_time = time.time()
            keypoints, scores = self.pose_model(upper_image, bboxes=upper_bboxes)
            lower_keypoints, lower_scores = self.pose_model(lower_image, bboxes=lower_bboxes)
            print(f"pose_time:{time.time() - start_time}s for {len(upper_bboxes) + len(lower_bboxes)} boxes")

            lower_keypoints[:, :, 1] = img_h - lower_keypoints[:, :, 1]
            lower_bboxes[:, [1, 3]] = img_h - lower_bboxes[:, [3, 1]]
            keypoints = np.vstack((keypoints, lower_keypoints))
            bboxes = np.vstack((upper_bboxes, lower_bboxes))
            scores = np.vstack((scores, lower_scores))
        
        return keypoints, scores, bboxes
    
    @staticmethod
    def format_result(keypoints_info: np.ndarray) -> List[PoseResult]:

        def format_keypoint_part(
                part: np.ndarray) -> Optional[List[Optional[Keypoint]]]:
            keypoints = [
                Keypoint(x, y, score, i) if score >= 0.3 else None
                for i, (x, y, score) in enumerate(part)
            ]
            return (None if all(keypoint is None
                                for keypoint in keypoints) else keypoints)

        def total_score(
                keypoints: Optional[List[Optional[Keypoint]]]) -> float:
            return (sum(
                keypoint.score for keypoint in keypoints
                if keypoint is not None) if keypoints is not None else 0.0)

        pose_results = []

        for instance in keypoints_info:
            body_keypoints = format_keypoint_part(
                instance[:18]) or ([None] * 18)
            left_hand = format_keypoint_part(instance[92:113])
            right_hand = format_keypoint_part(instance[113:134])
            face = format_keypoint_part(instance[24:92])

            # Openpose face consists of 70 points in total, while RTMPose only
            # provides 68 points. Padding the last 2 points.
            if face is not None:
                # left eye
                face.append(body_keypoints[14])
                # right eye
                face.append(body_keypoints[15])

            body = BodyResult(body_keypoints, total_score(body_keypoints),
                              len(body_keypoints))
            pose_results.append(PoseResult(body, left_hand, right_hand, face))

        return pose_results
