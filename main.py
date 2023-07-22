import numpy as np
import torch
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image(image_path):
    '''Loads an image from a file.'''
    return cv2.imread(image_path)

def convert_color_space(image):
    '''Converts an image from BGR to RGB color space.'''
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def generate_masks(image, model):
    '''Generates masks for an image using a SAM model.'''
    mask_generator = SamAutomaticMaskGenerator(model)
    return mask_generator.generate(image)

def overlay_masks_on_image(image, masks):
    '''Overlays masks on an image.'''
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(image_bgr, 0.7, masks, 0.3, 0)

def show_anns(anns):
    '''Creates an image from a list of masks.'''
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random(3)
        img[m] = color_mask
    return (img * 255).astype(np.uint8)


def main():
    # configs and paths
    model_type = "vit_h"
    checkpoint_relative_path = "sam_vit_h_4b8939.pth"
    image_relative_path = "images/blue-runners.jpg"

    logger.info("Starting script...")

    logger.info("Loading image...")
    image = load_image(image_relative_path)
    image = convert_color_space(image)

    logger.info("Loading SAM model...")
    model = sam_model_registry[model_type](checkpoint=checkpoint_relative_path)

    logger.info("Generating masks...")
    masks = generate_masks(image, model)
    mask_image = show_anns(masks)

    logger.info("Overlaying masks on image...")
    overlay = overlay_masks_on_image(image, mask_image)

    logger.info("Saving overlay image...")
    cv2.imwrite('images/overlay.jpg', overlay)

    logger.info("Script completed.")

if __name__ == "__main__":
    main()
