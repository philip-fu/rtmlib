from .object_detection import YOLOX, RTMDet, RTMDetRegional
from .pose_estimation import RTMO, RTMPose
from .solution import Body, Hand, PoseTracker, Wholebody, BodyWithFeet

__all__ = [
    'RTMDet', 'RTMPose', 'YOLOX', 'Wholebody', 'Body', 'Hand', 'PoseTracker',
    'RTMO', 'BodyWithFeet', 'RTMDetRegional'
]

import cv2
import numpy as np
from typing import List, Tuple, Optional
from shapely.geometry import Polygon
from shapely import minimum_bounding_circle
from shapely.affinity import translate, scale
import logging

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
    


#%%




# %%
def get_roundness(contour):
    area = cv2.contourArea(contour)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * radius**2
    if circle_area == 0:
        return 0
    return area / circle_area


def find_polygon(
    image: np.ndarray, # BGR
    label_point: List[int], # (y, x)
    lower_green: List[int] = [36, 50, 50],
    upper_green: List[int] = [86, 255, 255],
    max_correction_scale: float = 1.2, # relative scale,
    manual_shift_correction_in_x: int = 0,# in num of pixels,
    manual_shift_correction_in_y: int = 0,
    min_size: int = 50, # num pixels
    buffer_ratio: float = 0.2, # relative to image height
):
    if isinstance(lower_green, list):
        lower_green = np.array(lower_green)
    if isinstance(upper_green, list):
        upper_green = np.array(upper_green)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    num_labels, labels_im = cv2.connectedComponents(mask)

    # filter out small area
    for i in range(1, num_labels + 1):
        pts =  np.where(labels_im == i)
        if len(pts[0]) < min_size:
            labels_im[pts] = 0

    # find nearest
    nonzero = np.argwhere(labels_im != 0)
    distances = np.sqrt((nonzero[:,0] - label_point[0]) ** 2 + (nonzero[:,1] - label_point[1]) ** 2)
    nearest_index = np.argmin(distances)
    updated_label_point = nonzero[nearest_index]

    susan_mask = labels_im == labels_im[updated_label_point[0], updated_label_point[1]]

    # plt.imshow(labels_im)

    contours, _ = cv2.findContours(susan_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assuming only one object in the mask, take the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        polygon = Polygon(largest_contour[:, 0, :])

        # fill the gap
        if buffer_ratio > 0:
            buffer_size = image.shape[0] * 0.2
            polygon = polygon.buffer(buffer_size)
            polygon = polygon.buffer(-buffer_size)

        # corrections
        roundness_of_polygon = polygon.area / minimum_bounding_circle(polygon).area
        # angle = math.degrees(math.acos(roundness_of_polygon))
        
        # scale from btm point
        points = np.array(polygon.exterior.coords, np.int32)
        points = points.reshape((-1, 1, 2))
        bottom_point_of_circle = points[
            np.argmax(points[:, 0, 1]), 0, :
        ]
        top_point_of_circle = points[
            np.argmin(points[:, 0, 1]), 0, :
        ]
        correction_scale = (1 - roundness_of_polygon) * (max_correction_scale - 1) + 1
        print(f"Scaling circle by {correction_scale}.")
        # expand top
        polygon = scale(polygon, xfact=1, yfact=correction_scale, origin=tuple(bottom_point_of_circle.tolist()))
        # shrink btm
        polygon = scale(polygon, xfact=1, yfact=1. / correction_scale, origin=tuple(top_point_of_circle.tolist()))

        print(f"Shifting circle by ({manual_shift_correction_in_x}, {manual_shift_correction_in_y}).")
        moved_polygon = translate(polygon, xoff=manual_shift_correction_in_x, yoff=-manual_shift_correction_in_y)

        return moved_polygon
    else:
        return None


    
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def is_standing(
    keypoints, 
    standing_threshold_min: float = 150,
    standing_threshold_max: float = 180,
    check_leg_only: bool = True, # check hip-knee-ankle. otherwise check neck-hip-knee
) -> bool:
    """
    Checks if a person is standing based on OpenPose keypoints.
    True if either of the legs (hip-knee-ankle) is straight.

    Args:
        keypoints (list): List of keypoint coordinates (x, y) for a person.
                              Assumes keypoints are ordered as: 
                              [nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist, 
                               hip_r, hip_l, r_knee, l_knee, r_ankle, l_ankle, ...]
    Returns:
        bool: True if the person is standing, False otherwise.
    """
    if len(keypoints) < 14:
        return False
    
    if any([x is None for x in keypoints[8:14]]):
        return False

    if check_leg_only:
        left_leg_angle = calculate_angle(keypoints[8], keypoints[9], keypoints[10])
        right_leg_angle = calculate_angle(keypoints[11], keypoints[12], keypoints[13])
    else:
        left_leg_angle = calculate_angle(keypoints[1], keypoints[8], keypoints[9])
        right_leg_angle = calculate_angle(keypoints[1], keypoints[11], keypoints[12])

    is_standing = (standing_threshold_min <= left_leg_angle <= standing_threshold_max) or (standing_threshold_min <= right_leg_angle <= standing_threshold_max)

    logging.info(f"Angles of legs:\t{left_leg_angle}, {right_leg_angle}")

    return is_standing


def correct_far_end_standing_hands(
    bbox,
    circle_roundness: float = 0.65, # 1 means no correction needed. It's a proxy of the angle
    max_correction: float = 1., # move the box down by 1 box height
):
    """Aim to find its projection onto the table
    """
    projection_bbox = np.copy(bbox)
    bbox_height = bbox[3] - bbox[1]
    projection_bbox[[1,3]] += bbox_height * max_correction * (1 - circle_roundness)

    return projection_bbox