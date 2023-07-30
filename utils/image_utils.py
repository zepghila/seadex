import cv2
import os
import numpy as np
import logging
from .misc_utils import check_positive_integer

logger = logging.getLogger(__name__)

def load_image(image_path):
    '''Loads an image from a file.'''
    try:
        logger.info(f"Loading image from {image_path}...")
        return cv2.imread(image_path)
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise

def check_image_numpy(image):
    '''Checks if the image is a numpy array.'''
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy array.")

def convert_color_space_BGR2RGB(image):
    '''Converts an image from BGR to RGB color space.'''
    try:
        check_image_numpy(image)
        logger.info(f"Converting color space from BGR to RGB")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error converting color space: {e}")
        raise

def convert_color_space_RGB2BGR(image):
    '''Converts an image from RGB to BGR color space.'''
    try:
        check_image_numpy(image)
        logger.info(f"Converting color space from RGB to BGR")
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error converting color space: {e}")
        raise

def downsample_image_to_size(image, target_size_kb):
    '''Downsamples an image to a target file size in kilobytes.'''
    try:
        check_image_numpy(image)
        check_positive_integer(target_size_kb, 'target_size_kb')
        # Start with high quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        while True:
            # Encode image
            result, encimg = cv2.imencode('.jpg', image, encode_param)
            # Check size
            size_kb = len(encimg) / 1024
            if size_kb <= target_size_kb:
                break
            # Reduce quality
            encode_param[1] -= 10
            if encode_param[1] < 0:
                raise ValueError("Cannot reach the target size without completely degrading the image quality.")
        # Decode image
        image = cv2.imdecode(encimg, 1)
        return image
    except Exception as e:
        logger.error(f'Error downsampling image: {e}')
        raise
        
def scale_image_to_width(image, target_width):
    '''Scales an image to a target width, keeping aspect ratio.'''
    try:
        check_image_numpy(image)
        check_positive_integer(target_width, 'target_width')
        h, w = image.shape[:2]
        scaling_factor = target_width / float(w)
        target_height = int(h * scaling_factor)
        return cv2.resize(image, (target_width, target_height), interpolation = cv2.INTER_AREA)
    except Exception as e:
        logger.error(f'Error scaling image: {e}')
        raise

def scale_image_to_height(image, target_height):
    '''Scales an image to a target height, keeping aspect ratio.'''
    try:
        check_image_numpy(image)
        check_positive_integer(target_height, 'target_height') 
        h, w = image.shape[:2]
        scaling_factor = target_height / float(h)
        target_width = int(w * scaling_factor)
        return cv2.resize(image, (target_width, target_height), interpolation = cv2.INTER_AREA)
    except Exception as e:
        logger.error(f'Error scaling image: {e}')
        raise

def scale_image_to_size(image, target_width, target_height):
    '''Scales an image to a target size, keeping aspect ratio.'''
    try:
        check_image_numpy(image)
        check_positive_integer(target_width, 'target_width')
        check_positive_integer(target_height, 'target_height')
        h, w = image.shape[:2]
        scaling_factor = min(target_width / float(w), target_height / float(h))
        target_width = int(w * scaling_factor)
        target_height = int(h * scaling_factor)
        return cv2.resize(image, (target_width, target_height), interpolation = cv2.INTER_AREA)
    except Exception as e:
        logger.error(f'Error scaling image: {e}')
        raise

def draw_boxes_from_detections(image, detections):
    '''Draws bounding boxes on an image from a detections object'''
    #TODO mode into detection
    for detection in detections:
        class_id, cords, conf = detection
        x1, y1, x2, y2 = cords
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_id}: {conf}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image
