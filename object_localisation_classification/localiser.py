"""
Subsystem 1 - ROI Extraction
Uses OpenCV contour detection to crop the main object from an image.

Improved behavior:
- Handles small centered objects better than "largest contour only"
- Reduces background leakage on textured surfaces such as wood
- Returns a square crop centered around the detected object
- Returns original image if no reliable object is found
"""

from PIL import Image
import cv2
import numpy as np

PASSTHROUGH_AREA_RATIO = 0.85
SHRINK_RATIO = 0.06
SQUARE_PADDING_RATIO = 0.08
MIN_CONTOUR_AREA_RATIO = 0.001


def _is_already_cropped(box_xyxy, image_w, image_h):
    """Returns True if the detected box covers most of the image."""
    x1, y1, x2, y2 = box_xyxy
    box_area = (x2 - x1) * (y2 - y1)
    image_area = image_w * image_h
    return (box_area / image_area) >= PASSTHROUGH_AREA_RATIO


def _pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _score_contour(contour, image_w, image_h):
    """
    Score contour using:
    - larger area is better
    - closer to image center is better

    This works better than always taking the largest contour,
    especially for small centered objects on textured backgrounds.
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

    # Normalized center score: 1 near center, 0 far away
    center_score = 1.0 - (dist / max_dist)

    # Favor medium-large object contours but keep center importance high
    score = (area * 1.0) + (center_score * image_w * image_h * 0.05)

    return score


def _find_object_box(cv2_image: np.ndarray):
    """
    Finds the bounding box of the main object using thresholding + contour scoring.

    Returns (x1, y1, x2, y2) in pixel coordinates, or None if not found.
    """
    image_h, image_w = cv2_image.shape[:2]

    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try both normal and inverted threshold because object can be darker or lighter
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

            # Ignore contours that are almost the whole image
            if (w * h) > 0.95 * image_w * image_h:
                continue

            score = _score_contour(contour, image_w, image_h)

            if score > best_score:
                best_score = score
                best_contour = contour

    if best_contour is None:
        return None

    x, y, w, h = cv2.boundingRect(best_contour)
    return x, y, x + w, y + h


def _shrink_box(x1, y1, x2, y2, image_w, image_h):
    """
    Slightly shrink the detected bounding box to remove extra background.
    """
    shrink_x = int((x2 - x1) * SHRINK_RATIO)
    shrink_y = int((y2 - y1) * SHRINK_RATIO)

    x1 = min(image_w, x1 + shrink_x)
    y1 = min(image_h, y1 + shrink_y)
    x2 = max(0, x2 - shrink_x)
    y2 = max(0, y2 - shrink_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _make_square_box(x1, y1, x2, y2, image_w, image_h):
    """
    Convert rectangular object box into a square crop centered on the object.
    """
    box_w = x2 - x1
    box_h = y2 - y1

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    side = max(box_w, box_h)
    side = int(side * (1.0 + SQUARE_PADDING_RATIO))
    half = side / 2.0

    sq_x1 = int(round(cx - half))
    sq_y1 = int(round(cy - half))
    sq_x2 = int(round(cx + half))
    sq_y2 = int(round(cy + half))

    if sq_x1 < 0:
        shift = -sq_x1
        sq_x1 += shift
        sq_x2 += shift
    if sq_y1 < 0:
        shift = -sq_y1
        sq_y1 += shift
        sq_y2 += shift
    if sq_x2 > image_w:
        shift = sq_x2 - image_w
        sq_x1 -= shift
        sq_x2 -= shift
    if sq_y2 > image_h:
        shift = sq_y2 - image_h
        sq_y1 -= shift
        sq_y2 -= shift

    sq_x1 = max(0, sq_x1)
    sq_y1 = max(0, sq_y1)
    sq_x2 = min(image_w, sq_x2)
    sq_y2 = min(image_h, sq_y2)

    if sq_x2 <= sq_x1 or sq_y2 <= sq_y1:
        return None

    return sq_x1, sq_y1, sq_x2, sq_y2


def localise(pil_image: Image.Image, model=None) -> Image.Image:
    """
    Detect and crop the main object from pil_image.
    Returns a tighter, centered square crop when possible.
    Returns original if already cropped or no reliable object is found.
    """
    image_w, image_h = pil_image.size
    cv2_image = _pil_to_cv2(pil_image)

    box = _find_object_box(cv2_image)

    if box is None:
        return pil_image

    x1, y1, x2, y2 = box

    if _is_already_cropped((x1, y1, x2, y2), image_w, image_h):
        return pil_image

    shrunk = _shrink_box(x1, y1, x2, y2, image_w, image_h)
    if shrunk is None:
        return pil_image

    x1, y1, x2, y2 = shrunk

    square_box = _make_square_box(x1, y1, x2, y2, image_w, image_h)
    if square_box is None:
        return pil_image

    x1, y1, x2, y2 = square_box

    return pil_image.crop((x1, y1, x2, y2))


def load_localiser():
    """
    Kept for interface compatibility with prepare_image.py.
    OpenCV needs no model loading so this returns None.
    """
    return None
