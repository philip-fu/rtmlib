from typing import List, Tuple

import cv2
import numpy as np

from ..base import BaseTool
from .post_processings import nms, multiclass_nms
from ..pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine

class RTMDet(BaseTool):

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (640, 640),
                 mean: tuple = (103.5300, 116.2800, 123.6750),
                 std: tuple = (57.3750, 57.1200, 58.3950),
                 backend: str = 'onnxruntime',
                 device: str = 'cpu',
                 score_thr: float = 0.3,
                 nms_thr: float = 0.3,
                 ):
        super().__init__(onnx_model,
                         model_input_size,
                         mean,
                         std,
                         backend=backend,
                         device=device)
        self.score_thr = score_thr
        self.nms_thr = nms_thr

    def __call__(self, image: np.ndarray):
        image, ratio = self.preprocess(image)
        outputs = self.inference(image, dtype=np.float32)[0]
        results = self.postprocess(outputs, ratio)
        return results

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
        outputs: np.ndarray,
        ratio: float = 1.,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RTMDet model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMDet model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """

        if outputs.shape[-1] == 4:
            # onnx without nms module

            grids = []
            expanded_strides = []
            strides = [8, 16, 32]

            hsizes = [self.model_input_size[0] // stride for stride in strides]
            wsizes = [self.model_input_size[1] // stride for stride in strides]

            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                expanded_strides.append(np.full((*shape, 1), stride))

            grids = np.concatenate(grids, 1)
            expanded_strides = np.concatenate(expanded_strides, 1)
            outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
            outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

            predictions = outputs[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy,
                                  scores,
                                  nms_thr=self.nms_thr,
                                  score_thr=self.score_thr)
            if dets is not None:
                pack_dets = (dets[:, :4], dets[:, 4], dets[:, 5])
                final_boxes, final_scores, final_cls_inds = pack_dets
                isscore = final_scores > 0.3
                iscat = final_cls_inds == 0
                isbbox = [i and j for (i, j) in zip(isscore, iscat)]
                final_boxes = final_boxes[isbbox]
            else:
                final_boxes = np.empty((0, 4))

        elif outputs.shape[-1] == 5:

            pack_dets = (outputs[0, :, :4], outputs[0, :, 4])
            final_boxes, final_scores = pack_dets
            final_boxes /= ratio

            indices = final_scores >= self.score_thr
            final_boxes = final_boxes[indices]
            final_scores = final_scores[indices]
            indices = nms(
                boxes=final_boxes[:, :4],
                scores=final_scores,
                nms_thr=self.nms_thr,
            )

            final_boxes = np.hstack((final_boxes[indices], final_scores[indices].reshape((-1,1))))

        return final_boxes


class RTMDetRegional(BaseTool):

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (640, 640),
                 mean: tuple = (103.5300, 116.2800, 123.6750),
                 std: tuple = (57.3750, 57.1200, 58.3950),
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        super().__init__(onnx_model,
                         model_input_size,
                         mean,
                         std,
                         backend=backend,
                         device=device)

    def __call__(self, image: np.ndarray, bboxes: list = []):
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        result_bboxes = []
        for bbox in bboxes:
            img_crop, center, scale = self.preprocess(image, bbox)
            outputs = self.inference(img_crop, dtype=np.float32)[0]

            result_bbox = self.postprocess(outputs, center, scale)

            result_bboxes.append(result_bbox)

        result_bboxes = np.concatenate(result_bboxes, axis=0)

        return result_bboxes, []

    def preprocess(self, img: np.ndarray, bbox: list):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        bbox = np.array(bbox)

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine(self.model_input_size, scale,
                                             center, img)

        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            resized_img = (resized_img - self.mean) / self.std

        return resized_img, center, scale

    def postprocess(
        self,
        outputs: np.ndarray,
        center: Tuple[int, int],
        scale: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RTMDet model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMDet model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """

        if outputs.shape[-1] == 4:
            # onnx without nms module

            grids = []
            expanded_strides = []
            strides = [8, 16, 32]

            hsizes = [self.model_input_size[0] // stride for stride in strides]
            wsizes = [self.model_input_size[1] // stride for stride in strides]

            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                expanded_strides.append(np.full((*shape, 1), stride))

            grids = np.concatenate(grids, 1)
            expanded_strides = np.concatenate(expanded_strides, 1)
            outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
            outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

            predictions = outputs[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy = self.rescale_bboxes(bboxes=boxes_xyxy, center=center, scale=scale)
            dets = multiclass_nms(boxes_xyxy,
                                  scores,
                                  nms_thr=self.nms_thr,
                                  score_thr=self.score_thr)
            if dets is not None:
                pack_dets = (dets[:, :4], dets[:, 4], dets[:, 5])
                final_boxes, final_scores, final_cls_inds = pack_dets
                isscore = final_scores > 0.3
                iscat = final_cls_inds == 0
                isbbox = [i and j for (i, j) in zip(isscore, iscat)]
                final_boxes = final_boxes[isbbox]

        elif outputs.shape[-1] == 5:
            # onnx contains nms module

            pack_dets = (outputs[0, :, :4], outputs[0, :, 4])
            final_boxes, final_scores = pack_dets
            final_boxes = self.rescale_bboxes(bboxes=final_boxes, center=center, scale=scale)

        return final_boxes
    
    def rescale_bboxes(
        self,
        bboxes: np.ndarray,
        center: Tuple[int, int],
        scale: Tuple[int, int],
    ) -> np.ndarray:
        """bboxes of shape (N, 4), xyxy
        From RTMPose.postprocess
        """
        bboxes[:, [0, 1]] = bboxes[:, [0, 1]] / self.model_input_size * scale
        bboxes[:, [2, 3]] = bboxes[:, [2, 3]] / self.model_input_size * scale
        bboxes[:, [0, 1]] = bboxes[:, [0, 1]] + center - scale / 2
        bboxes[:, [2, 3]] = bboxes[:, [2, 3]] + center - scale / 2

        return bboxes