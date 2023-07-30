import logging

import cv2

from config import sam_model_type, sam_model_checkpoint_relative_path, input_image_relative_path, default_max_image_width
from segmentation_tools import segment_SAM
from object_detection_tools import object_detect_yolov8
from utils import load_image, convert_color_space_BGR2RGB, scale_image_to_width, draw_boxes_from_detections



# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting script...")

        image = load_image(input_image_relative_path)
        image = convert_color_space_BGR2RGB(image)
        image = scale_image_to_width(image, default_max_image_width)

        masks_image, segmented_image = segment_SAM(image, sam_model_type, sam_model_checkpoint_relative_path) 
        
        logger.info("Saving masks and segmented images...")
        cv2.imwrite('images/masks.jpg', masks_image)
        cv2.imwrite('images/segmented.jpg', segmented_image)

        image = segmented_image

        detections = object_detect_yolov8(image)
        if detections is not None:
            boxed_image = draw_boxes_from_detections(image, detections)
            cv2.imwrite('images/detected.jpg', boxed_image)
        else:
            logger.info("No detections were made.")

        logger.info("Script completed.")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
