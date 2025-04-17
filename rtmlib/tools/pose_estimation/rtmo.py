from typing import List, Tuple

import cv2
import numpy as np

from ..base import BaseTool
from .post_processings import convert_coco_to_openpose

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

class RTMO(BaseTool):

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (640, 640),
                 score_threshold: float = 0.3,
                 mean: tuple = None,
                 std: tuple = None,
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        super().__init__(onnx_model, model_input_size, mean, std, backend,
                         device)
        self.to_openpose = to_openpose
        self.score_threshold = score_threshold

    def __call__(self, image: np.ndarray):
        image, ratio = self.preprocess(image)
        outputs = self.inference(image)

        keypoints, scores = self.postprocess(outputs, ratio)

        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        return keypoints, scores

    def preprocess(self, img: np.ndarray):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        if len(img.shape) == 3:
            padded_img = np.ones(
                (self.model_input_size[0], self.model_input_size[1], 3),
                dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.model_input_size, dtype=np.uint8) * 114

        ratio = min(self.model_input_size[0] / img.shape[0],
                    self.model_input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            padded_img = (padded_img - self.mean) / self.std

        return padded_img, ratio

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float = 1.,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RTMO model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMO model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """
        det_outputs, pose_outputs = outputs

        # onnx contains nms module
        pack_dets = (det_outputs[0, :, :4], det_outputs[0, :, 4])
        final_boxes, final_scores = pack_dets
        final_boxes /= ratio
        isscore = final_scores > self.score_threshold
        isbbox = [i for i in isscore]
        # final_boxes = final_boxes[isbbox]

        # decode pose outputs
        keypoints, scores = pose_outputs[0, :, :, :2], pose_outputs[0, :, :, 2]
        keypoints = keypoints / ratio

        keypoints = keypoints[isbbox]
        scores = scores[isbbox]

        return keypoints, scores
    
    @staticmethod
    def transform_keypoints_to_roi(
        keypoints: np.ndarray,
        no_man_area: List[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Infer roi from openpose 18 keypoints.
        Return a tuple of (head boxes, hand boxes)
        """
        head_keypoint_indexes = [0, 1, 2, 5, 14, 15, 16, 17]
        upper_body_keypoint_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        left_forearm_keypoint_indexes = [6, 7]
        right_forearm_keypoint_indexes = [3, 4]
        
        head_bboxes = np.vstack((
            keypoints[:, head_keypoint_indexes, 0].min(axis=1),
            keypoints[:, head_keypoint_indexes, 1].min(axis=1),
            keypoints[:, head_keypoint_indexes, 0].max(axis=1),
            keypoints[:, head_keypoint_indexes, 1].max(axis=1),
        )).transpose((1, 0))

        person_bboxes = np.vstack((
            keypoints[:, upper_body_keypoint_indexes, 0].min(axis=1),
            keypoints[:, upper_body_keypoint_indexes, 1].min(axis=1),
            keypoints[:, upper_body_keypoint_indexes, 0].max(axis=1),
            keypoints[:, upper_body_keypoint_indexes, 1].max(axis=1),
        )).transpose((1, 0))

        selected_idxs = []
        if no_man_area is not None:
            for idx, bbox in enumerate(person_bboxes):
                bbox_overlap_with_no_man_area = bb_intersection_over_boxB(no_man_area, bbox)
                if bbox_overlap_with_no_man_area <= 0.8:
                    selected_idxs.append(idx)

            head_bboxes = head_bboxes[selected_idxs]
            keypoints = keypoints[selected_idxs, :, :]

        # approximate hand size with 0.5 * forearm
        hand_sizes = (np.hstack((
            np.linalg.norm(keypoints[:, left_forearm_keypoint_indexes[0], :] - keypoints[:, left_forearm_keypoint_indexes[1], :], axis=1),
            np.linalg.norm(keypoints[:, right_forearm_keypoint_indexes[0], :] - keypoints[:, right_forearm_keypoint_indexes[1], :], axis=1),
        )) * 0.5).reshape((-1, 1))

        hand_tan_values = np.hstack((
            (
                keypoints[:, left_forearm_keypoint_indexes[0], 1] \
                    - keypoints[:, left_forearm_keypoint_indexes[1], 1]
            ) / (
                keypoints[:, left_forearm_keypoint_indexes[0], 0] \
                    - keypoints[:, left_forearm_keypoint_indexes[1], 0]
            ),
            (
                keypoints[:, right_forearm_keypoint_indexes[0], 1] \
                    - keypoints[:, right_forearm_keypoint_indexes[1], 1]
            ) / (
                keypoints[:, right_forearm_keypoint_indexes[0], 0] \
                    - keypoints[:, right_forearm_keypoint_indexes[1], 0]
            )
        ))

        hand_radians = np.arctan(hand_tan_values)
        hand_sin_values = np.sin(hand_radians)
        hand_cos_values = np.cos(hand_radians)
        
        hand_wrists = np.vstack((
            keypoints[:, left_forearm_keypoint_indexes[1], :],
            keypoints[:, right_forearm_keypoint_indexes[1], :]
        ))

        hand_tips = hand_wrists + \
            hand_sizes * \
                np.vstack((
            hand_cos_values, hand_sin_values
        )).transpose((1, 0)) # (x, y) + distance * (cos, sin)

        hand_ends = hand_wrists - \
            hand_sizes * \
                np.vstack((
            hand_cos_values, hand_sin_values
        )).transpose((1, 0)) # (x, y) - distance * (cos, sin)

        hand_xs = np.vstack((
            hand_tips[:, 0], hand_ends[:, 0]
        )).transpose((1, 0))

        hand_ys = np.vstack((
            hand_tips[:, 1], hand_ends[:, 1]
        )).transpose((1, 0))
        
        hand_bboxes = np.vstack((
            hand_xs.min(axis=1),
            hand_ys.min(axis=1),
            hand_xs.max(axis=1),
            hand_ys.max(axis=1),
        )).transpose((1, 0))

        # make sure hand boxes are not too narrow.
        hand_bboxes_widths = (hand_bboxes[:, 2] - hand_bboxes[:, 0]).reshape((-1, 1))
        hand_bboxes_heights = (hand_bboxes[:, 3] - hand_bboxes[:, 1]).reshape((-1, 1))
        hand_bboxes_center_xs = ((hand_bboxes[:, 2] + hand_bboxes[:, 0]) / 2).reshape((-1, 1))
        hand_bboxes_center_ys = ((hand_bboxes[:, 3] + hand_bboxes[:, 1]) / 2).reshape((-1, 1))

        hand_bboxes_widths = np.max((hand_bboxes_widths, hand_sizes), axis=0)
        hand_bboxes_heights = np.max((hand_bboxes_heights, hand_sizes), axis=0)
        
        hand_bboxes = np.hstack((
            hand_bboxes_center_xs - hand_bboxes_widths / 2,
            hand_bboxes_center_ys - hand_bboxes_heights / 2,
            hand_bboxes_center_xs + hand_bboxes_widths / 2,
            hand_bboxes_center_ys + hand_bboxes_heights / 2,
        ))

        return head_bboxes, hand_bboxes, selected_idxs
