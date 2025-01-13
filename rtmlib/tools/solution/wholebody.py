import time
from typing import List, Optional
import cv2
import numpy as np
import logging
from .. import YOLOX, RTMPose, RTMDet, RTMDetRegional
from .utils.types import BodyResult, Keypoint, PoseResult

def bb_intersection_over_boxB(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / boxBArea
    # return the intersection over union value
    return iou

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
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_s_8xb8-300e_humanart-3ef259a7.zip',  # noqa
            'det_input_size': (640, 640),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip',  # noqa
            'pose_input_size': (192, 256),
        },
        'lightweight_rtm': {
            'det':
            'https://mmdeploy-oss.openmmlab.com/model/mmpose/rtmdet-37adb8.onnx',  # noqa
            'det_input_size': (320, 320),
            'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.zip',
            # "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip",
            # 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.zip',
            # 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.zip',  # noqa
            'pose_input_size': (192, 256),
            'pose_heavy': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip',
            # "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip",
            # 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.zip',
            # 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.zip',  # noqa
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
                 score_thres: float = 0.3,
                 nms_thres: float = 0.3,
                 device: str = 'cpu'):
        
        self.mode = mode

        if det is None:
            det = self.MODE[mode]['det']
            det_input_size = self.MODE[mode]['det_input_size']

        if pose is None:
            pose = self.MODE[mode]['pose']
            pose_heavy = self.MODE[mode].get('pose_heavy')
            pose_input_size = self.MODE[mode]['pose_input_size']

        if 'rtm' in mode:
            self.do_flip = True
            self.det_model = RTMDet(det,
                                model_input_size=det_input_size,
                                backend=backend,
                                device=device,
                                score_thr=score_thres,
                                nms_thr=nms_thres)
        else:
            self.do_flip = False
            self.det_model = YOLOX(det,
                                model_input_size=det_input_size,
                                backend=backend,
                                device=device,
                                score_thr=score_thres,
                                nms_thr=nms_thres)
            
        self.det_model.score_thr = score_thres
        
        self.pose_model = RTMPose(pose,
                                model_input_size=pose_input_size,
                                to_openpose=to_openpose,
                                backend=backend,
                                device=device)
        if pose_heavy is not None:
            self.num_boxes_to_use_heavy = 10
            self.pose_model_heavy = RTMPose(pose_heavy,
                                model_input_size=pose_input_size,
                                to_openpose=to_openpose,
                                backend=backend,
                                device=device,
                                padding=2)
        else:
            self.num_boxes_to_use_heavy = -1
            self.pose_model_heavy = None

    def filter_bbox(
        self,
        bboxes: np.ndarray,
        no_man_area: List[int]
    ) -> np.ndarray:
        selected_idxs = []
        for idx, bbox in enumerate(bboxes):
            bbox_overlap_with_no_man_area = bb_intersection_over_boxB(no_man_area, bbox)
            if bbox_overlap_with_no_man_area <= 0.7:
                selected_idxs.append(idx)

        return bboxes[selected_idxs]

    def __call__(self, image: np.ndarray, no_man_area: Optional[List[int]]=None):
        """One inference for upper image (with some buffer). One for lower.
        WARNING: there is no dedup here.

        Use a bigger pose model when # boxes are low
        """
        if not self.do_flip:
            start_time = time.time()
            bboxes = self.det_model(image)
            if no_man_area is not None:
                bboxes = self.filter_bbox(bboxes=bboxes, no_man_area=no_man_area)
            
            logging.info(f"det_time:{time.time() - start_time}s")
            start_time = time.time()
            if len(bboxes) <= self.num_boxes_to_use_heavy:
                keypoints, scores = self.pose_model_heavy(image, bboxes=bboxes[:,:4])
            else:
                keypoints, scores = self.pose_model(image, bboxes=bboxes[:,:4])
            logging.info(f"pose_time:{time.time() - start_time}s for {len(bboxes)} boxes")
        else:
            img_h, img_w, _ =  image.shape
            upper_image = np.copy(image)
            upper_image[int(img_h / 2 * 1.2):, :] = 255.
            lower_image = cv2.flip(image, 0)
            lower_image[int(img_h / 2 * 1.2):, :] = 255.

            start_time = time.time()
            upper_bboxes = self.det_model(upper_image)
            lower_bboxes = self.det_model(lower_image)
            logging.info(f"det_time:{time.time() - start_time}s")
            start_time = time.time()

            if no_man_area is not None:
                upper_bboxes = self.filter_bbox(bboxes=upper_bboxes, no_man_area=no_man_area)
                no_man_area_flip = np.copy(no_man_area)
                no_man_area_flip[[1,3]] = img_h - no_man_area_flip[[3,1]]
                lower_bboxes = self.filter_bbox(bboxes=lower_bboxes, no_man_area=no_man_area_flip)
            
            if len(upper_bboxes) + len(lower_bboxes) <= self.num_boxes_to_use_heavy:
                keypoints, scores = self.pose_model_heavy(upper_image, bboxes=upper_bboxes[:,:4])
                lower_keypoints, lower_scores = self.pose_model_heavy(lower_image, bboxes=lower_bboxes[:,:4])
            else:
                keypoints, scores = self.pose_model(upper_image, bboxes=upper_bboxes[:,:4])
                lower_keypoints, lower_scores = self.pose_model(lower_image, bboxes=lower_bboxes[:,:4])
            
            logging.info(f"pose_time:{time.time() - start_time}s for {len(upper_bboxes) + len(lower_bboxes)} boxes")

            lower_keypoints[:, :, 1] = img_h - lower_keypoints[:, :, 1]
            lower_bboxes[:, [1, 3]] = img_h - lower_bboxes[:, [3, 1]]

            if len(keypoints) == 0:
                keypoints = lower_keypoints
                scores = lower_scores
            elif len(lower_keypoints) == 0:
                pass
            else:
                keypoints = np.vstack((keypoints, lower_keypoints))
                scores = np.vstack((scores, lower_scores))

            bboxes = np.vstack((upper_bboxes, lower_bboxes))
            
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
