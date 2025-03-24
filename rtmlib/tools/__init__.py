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
    


#%%
import cv2
import numpy as np
from typing import List
from shapely.geometry import Polygon
from shapely import minimum_bounding_circle
from shapely.affinity import translate, scale



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
    label_point: List[int],
    lower_green: List[int] = [36, 50, 50],
    upper_green: List[int] = [86, 255, 255],
    max_correction_scale: float = 1.2, # relative scale,
    manual_shift_correction_in_x: int = 0,# in num of pixels,
    manual_shift_correction_in_y: int = 0, 
):
    if isinstance(lower_green, list):
        lower_green = np.array(lower_green)
    if isinstance(upper_green, list):
        upper_green = np.array(upper_green)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    num_labels, labels_im = cv2.connectedComponents(mask)
    susan_mask = labels_im == labels_im[label_point[0], label_point[1]]

    # plt.imshow(labels_im)

    contours, _ = cv2.findContours(susan_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assuming only one object in the mask, take the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        polygon = Polygon(largest_contour[:, 0, :])

        # corrections
        roundness_of_polygon = polygon.area / minimum_bounding_circle(polygon).area
        # angle = math.degrees(math.acos(roundness_of_polygon))
        
        # scale from btm point
        points = np.array(polygon.exterior.coords, np.int32)
        points = points.reshape((-1, 1, 2))
        bottom_point_of_circle = points[
            np.argmax(points[:, 0, 1]), 0, :
        ]
        correction_scale = (1 - roundness_of_polygon) * (max_correction_scale - 1) + 1
        print(f"Scaling circle by {correction_scale} originating from bottom.")
        polygon = scale(polygon, xfact=1, yfact=correction_scale, origin=tuple(bottom_point_of_circle.tolist()))

        print(f"Shifting circle by ({manual_shift_correction_in_x}, {manual_shift_correction_in_y}).")
        moved_polygon = translate(polygon, xoff=manual_shift_correction_in_x, yoff=-manual_shift_correction_in_y)

        return moved_polygon
    else:
        return None
