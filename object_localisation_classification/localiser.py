"""
Subsystem 1 - ROI Extraction
Uses OpenCV contour detection to crop the main object from an image.

Behavior:
- Rectangular objects are cropped exactly to their detected rectangular bounds
- Non-rectangular objects receive only a small safety margin
- Avoids harsh cutting of object edges
- If the object already occupies almost the entire image, the original image is returned
- Returns original image if no reliable object is found
"""

from PIL import Image
import cv2
import numpy as np


# If the detected object covers this much of the image, assume the image is
# already effectively cropped and return it unchanged.
PASSTHROUGH_AREA_RATIO = 0.92

# Border tolerance used when checking whether the object already touches
# the image edges and therefore has little or no background.
EDGE_TOLERANCE_RATIO = 0.03

# Minimum contour area relative to total image area.
MIN_CONTOUR_AREA_RATIO = 0.001

# Small padding for irregular objects only.
IRREGULAR_PADDING_RATIO = 0.04

# A contour is considered rectangular if it fills most of its bounding box.
RECT_FILL_RATIO_THRESHOLD = 0.90

# Approximated polygon should have around 4 corners for a rectangle.
POLYGON_RECTANGULARITY_THRESHOLD = 4


def _pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL image to OpenCV BGR array.
    """
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _contour_to_box(contour):
    """
    Return contour bounding box as (x1, y1, x2, y2).
    """
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, x + w, y + h


def _is_already_cropped(box_xyxy, image_w, image_h):
    """
    Returns True if the detected object already occupies almost the whole image.

    This protects cases where:
    - the image already contains only the object
    - wood / texture / surface fills the full frame
    - there is no meaningful background to remove

    Logic:
    - if bounding box covers most of the image area
    - OR if bounding box nearly touches all image borders
    """
    x1, y1, x2, y2 = box_xyxy

    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    image_area = image_w * image_h

    if image_area <= 0:
        return True

    area_ratio = box_area / image_area
    covers_area = area_ratio >= PASSTHROUGH_AREA_RATIO

    tol_x = int(round(image_w * EDGE_TOLERANCE_RATIO))
    tol_y = int(round(image_h * EDGE_TOLERANCE_RATIO))

    touches_edges = (
        x1 <= tol_x and
        y1 <= tol_y and
        x2 >= (image_w - tol_x) and
        y2 >= (image_h - tol_y)
    )

    return covers_area or touches_edges


def _score_contour(contour, image_w, image_h):
    """
    Score contour using:
    - larger area is better
    - closer to image center is better
    """
    area = cv2.contourArea(contour)
    if area <= 0:
        return -1e9

    x, y, w, h = cv2.boundingRect(contour)

    cx = x + w / 2.0
    cy = y + h / 2.0

    img_cx = image_w / 2.0
    img_cy = image_h / 2.0

    dist = np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
    max_dist = np.sqrt(img_cx ** 2 + img_cy ** 2)

    if max_dist == 0:
        center_score = 1.0
    else:
        center_score = 1.0 - (dist / max_dist)

    score = (area * 1.0) + (center_score * image_w * image_h * 0.05)
    return score


def _find_object_contour(cv2_image: np.ndarray):
    """
    Finds the main object contour using thresholding and contour scoring.

    Returns:
    - best contour if found
    - None otherwise
    """
    image_h, image_w = cv2_image.shape[:2]

    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    threshold_candidates = []

    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold_candidates.append(thresh)

    _, thresh_inv = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    threshold_candidates.append(thresh_inv)

    best_contour = None
    best_score = -1e9
    min_area = image_w * image_h * MIN_CONTOUR_AREA_RATIO

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    for binary in threshold_candidates:
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Skip contours that are almost full image.
            # These often represent already-cropped surfaces or frame-wide regions.
            if (w * h) > 0.98 * image_w * image_h:
                continue

            score = _score_contour(contour, image_w, image_h)

            if score > best_score:
                best_score = score
                best_contour = contour

    return best_contour


def _is_rectangular_shape(contour):
    """
    Decide whether the contour is truly rectangular.

    Conditions:
    - contour fills most of its bounding rectangle
    - approximated contour has about 4 corners
    """
    x, y, w, h = cv2.boundingRect(contour)

    if w <= 0 or h <= 0:
        return False

    rect_area = w * h
    contour_area = cv2.contourArea(contour)

    if rect_area <= 0 or contour_area <= 0:
        return False

    fill_ratio = contour_area / rect_area

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False

    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    return (
        fill_ratio >= RECT_FILL_RATIO_THRESHOLD and
        len(approx) <= POLYGON_RECTANGULARITY_THRESHOLD
    )


def _pad_box(x1, y1, x2, y2, image_w, image_h, padding_ratio):
    """
    Expand a box by a small padding ratio.
    """
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(round(box_w * padding_ratio))
    pad_y = int(round(box_h * padding_ratio))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image_w, x2 + pad_x)
    y2 = min(image_h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _build_final_crop_box(contour, image_w, image_h):
    """
    Build final crop box.

    - Rectangular objects: exact crop, no padding
    - Other objects: slight padding so edges are not cut
    """
    x1, y1, x2, y2 = _contour_to_box(contour)

    if _is_rectangular_shape(contour):
        return x1, y1, x2, y2

    return _pad_box(x1, y1, x2, y2, image_w, image_h, IRREGULAR_PADDING_RATIO)


def localise(pil_image: Image.Image, model=None) -> Image.Image:
    """
    Detect and crop the main object from pil_image.

    Behavior:
    - rectangular objects are cropped exactly
    - irregular objects are cropped tightly with small safety margin
    - if the detected object already fills the image, the original image is returned
    - returns original image if no reliable object is found
    """
    image_w, image_h = pil_image.size
    cv2_image = _pil_to_cv2(pil_image)

    contour = _find_object_contour(cv2_image)

    if contour is None:
        return pil_image

    raw_box = _contour_to_box(contour)

    # Do not crop if image is already basically just the object.
    if _is_already_cropped(raw_box, image_w, image_h):
        return pil_image

    final_box = _build_final_crop_box(contour, image_w, image_h)

    if final_box is None:
        return pil_image

    x1, y1, x2, y2 = final_box
    return pil_image.crop((x1, y1, x2, y2))


def load_localiser():
    """
    Kept for interface compatibility with prepare_image.py.
    OpenCV needs no model loading so this returns None.
    """
    return None