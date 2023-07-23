from config import sam_model_type, sam_model_checkpoint_relative_path, input_image_relative_path
from segmentation_tools import load_sam_model, generate_masks, show_anns
from utils.image_utils import load_image, convert_color_space_BGR2RGB, overlay_masks_on_image
import cv2
import logging

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting script...")

        image = load_image(input_image_relative_path)
        image = convert_color_space_BGR2RGB(image)

        model = load_sam_model(sam_model_type, sam_model_checkpoint_relative_path)

        masks = generate_masks(image, model)
        mask_image = show_anns(masks)
        
        cv2.imwrite('images/masks.jpg', mask_image)

        overlay = overlay_masks_on_image(image, mask_image)

        logger.info("Saving overlay image...")
        cv2.imwrite('images/overlay.jpg', overlay)

        logger.info("Script completed.")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
