from .object_detection import YOLOX, RTMDet, RTMDetRegional
from .pose_estimation import RTMO, RTMPose
from .solution import Body, Hand, PoseTracker, Wholebody, BodyWithFeet

__all__ = [
    'RTMDet', 'RTMPose', 'YOLOX', 'Wholebody', 'Body', 'Hand', 'PoseTracker',
    'RTMO', 'BodyWithFeet', 'RTMDetRegional'
]

import cv2
import numpy as np
from typing import Tuple, Optional


def find_susan(
    image: np.ndarray,
    scale: int = 1,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Input image bgr. 
    Return (x, y, r * scale) of the circle closest to the center of the image.
    Return tuple of None's in the same shape if circle not found.
    """

    scale = 1.0

    image_size = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

    # detect circles in the image
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=70)
    # ensure at least some circles were found

    # print(circles)
    if circles is not None:
        circles = circles[:, :2, :]
        distances_to_center = np.sum(np.abs(circles[0, :, :2] - np.array([image_size[1]/2, image_size[0]/2])) ** 2, axis=-1) ** (1/2.)

        selected_circle_idx_1, selected_circle_idx_2 = None, None
        min_dist_1, min_dist_2 = np.inf, np.inf
        for idx, distance_to_center in enumerate(distances_to_center):
            if distance_to_center < min_dist_1:
                if min_dist_1 < min_dist_2:
                    min_dist_2 = min_dist_1
                    selected_circle_idx_2 = selected_circle_idx_1

                min_dist_1 = distance_to_center
                selected_circle_idx_1 = idx
            elif distance_to_center < min_dist_2:
                min_dist_2 = distance_to_center
                selected_circle_idx_2 = idx


        if selected_circle_idx_1 is not None:
            if selected_circle_idx_2 is not None:
                if circles[0, selected_circle_idx_1, 2] < circles[0, selected_circle_idx_2, 2]:
                    selected_circle_idx = selected_circle_idx_1
                else:
                    selected_circle_idx = selected_circle_idx_2
            else:
                selected_circle_idx = selected_circle_idx_1
        
        # convert the (x, y) coordinates and radius of the circles to integers

        (x, y, r) = np.round(circles[0, selected_circle_idx]).astype("int")
        
        # make larger
        r *= scale
        r = int(r)

        return (x, y, r)
    else:
        print("No circle found.")
        return (None, None, None)