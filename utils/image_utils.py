import cv2
import logging

logger = logging.getLogger(__name__)

def load_image(image_path):
    '''Loads an image from a file.'''
    try:
        logger.info(f"Loading image from {image_path}...")
        return cv2.imread(image_path)
    except Exception as e:
        logger.error(f"Error laoding image: {e}")
        raise

def convert_color_space_BGR2RGB(image):
    '''Converts an image from BGR to RGB color space.'''
    try:
        logger.info(f"Converting color space from BGR to RGB")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error converting color space: {e}")
        raise

def convert_color_space_RGB2BGR(image):
    '''Converts an image from BGR to RGB color space.'''
    try:
        logger.info(f"Converting color space from RGB to BGR")
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error converting color space: {e}")
        raise

def overlay_masks_on_image(image, masks):
    '''Overlays masks on an image.'''
    try:
        logger.info("Overlaying masks on image...")
        image_bgr = convert_color_space_RGB2BGR(image)
        return cv2.addWeighted(image_bgr, 0.7, masks, 0.3, 0)
    except Exception as e:
        logger.error(f"Error overlaying masks on image: {e}")
        raise
