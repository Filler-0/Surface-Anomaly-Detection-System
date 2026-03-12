"""
Subsystem 1 - ROI Extraction
Uses OpenCV contour detection to crop the main object from an image.
Assumes one object per image (industrial inspection context).
If the object already fills >75% of the image it is returned as-is.
"""

from PIL import Image
import cv2
import numpy as np

PASSTHROUGH_AREA_RATIO = 0.75
PADDING_RATIO          = 0.05


def _is_already_cropped(box_xyxy, image_w, image_h):
    """Returns True if the detected box covers most of the image."""
    x1, y1, x2, y2 = box_xyxy
    box_area   = (x2 - x1) * (y2 - y1)
    image_area = image_w * image_h
    return (box_area / image_area) >= PASSTHROUGH_AREA_RATIO


def _pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _find_object_box(cv2_image: np.ndarray):
    """
    Finds the bounding box of the main object using contour detection.
    Works well for MVTec-style images with plain/uniform backgrounds.

    Returns (x1, y1, x2, y2) in pixel coordinates, or None if not found.
    """
    gray    = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu thresholding - automatically finds best threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to clean up noise and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Try inverted threshold - object might be darker than background
        _, thresh_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh_inv = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel)
        thresh_inv = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN,  kernel)
        contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Take the largest contour - that is the main object
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, x + w, y + h


def localise(pil_image: Image.Image, model=None) -> Image.Image:
    """
    Detect and crop the main object from pil_image.
    The model parameter is kept for interface compatibility but is not used.
    Returns cropped PIL image, or original if already cropped or no object found.
    """
    image_w, image_h = pil_image.size
    cv2_image = _pil_to_cv2(pil_image)

    box = _find_object_box(cv2_image)

    if box is None:
        # No object found - return original
        return pil_image

    x1, y1, x2, y2 = box

    # If object already fills most of the image - no need to crop
    if _is_already_cropped((x1, y1, x2, y2), image_w, image_h):
        return pil_image

    # Add padding around the detected box
    pad_x = int((x2 - x1) * PADDING_RATIO)
    pad_y = int((y2 - y1) * PADDING_RATIO)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image_w, x2 + pad_x)
    y2 = min(image_h, y2 + pad_y)

    return pil_image.crop((x1, y1, x2, y2))


def load_localiser():
    """
    Kept for interface compatibility with prepare_image.py.
    OpenCV needs no model loading so this returns None.
    """
    return None