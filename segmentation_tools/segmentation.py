from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_sam_model(model_type, checkpoint_relative_path):
    '''Loads a SAM model.'''
    try:
        logger.info("Loading SAM model...")
        model = sam_model_registry[model_type](checkpoint=checkpoint_relative_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
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
