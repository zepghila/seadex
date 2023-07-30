from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from utils import check_image_numpy, convert_color_space_RGB2BGR
import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)

def load_sam_model(model_type, checkpoint_relative_path):
    '''Loads a SAM model.'''
    try:
        logger.info("Loading SAM model...")
        model = sam_model_registry[model_type](checkpoint=checkpoint_relative_path)
        return model
    except Exception as e:
        logger.error(f"Error loading SAM model: {e}")
        raise

def generate_masks(image, model):
    '''Generates masks for an image using a SAM model.'''
    try:
        logger.info("Generating masks...")
        mask_generator = SamAutomaticMaskGenerator(model)
        return mask_generator.generate(image)
    except Exception as e:
        logger.error(f"Error generating masks: {e}")
        raise

def show_anns(anns):
    '''Creates an image from a list of masks.'''
    try:
        logger.info("Creating image from masks...")
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.random.random(3)
            img[m] = color_mask
        return (img * 255).astype(np.uint8)
    except Exception as e:
        logger.error(f"Error creating image from masks: {e}")
        raise

def overlay_segmentation_masks_on_image(image, masks):
    '''Overlays masks on an image.'''
    try:
        check_image_numpy(image)
        logger.info("Overlaying masks on image...")
        image_bgr = convert_color_space_RGB2BGR(image)
        return cv2.addWeighted(image_bgr, 0.7, masks, 0.3, 0)
    except Exception as e:
        logger.error(f"Error overlaying masks on image: {e}")
        raise

def segment_SAM(image, sam_model_type, sam_model_checkpoint_relative_path):
    '''Segment input image using Segment Anything Model (SAM)'''
    try:
        logger.info("Starting Segmentation using SAM...")
        model = load_sam_model(sam_model_type, sam_model_checkpoint_relative_path)
        masks = generate_masks(image, model)
        mask_image = show_anns(masks)
        segmented_image = overlay_segmentation_masks_on_image(image, mask_image)
        return mask_image, segmented_image
    except Exception as e:
        logger.error(f"Error segmenting image using SAM: {e}")
        raise