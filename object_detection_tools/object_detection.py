import logging

from ultralytics import YOLO

from config import yolo_model_relative_path
from utils import load_image



logger = logging.getLogger(__name__)

def load_yolo_model():
    '''Loads a Yolo model.'''
    try:
        logger.info("Loading Yolo model...")
        model = YOLO('yolov8m.pt')  # load an official model
        return model
    except Exception as e:
        logger.error(f"Error loading Yolo model: {e}")
        raise

def create_detections(result):
    '''Creates a list of detections from a YOLO result.'''
    detections = []
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        detections.append((class_id, cords, conf))
    return detections

def object_detect_yolov8(image):
    '''Runs a Yolo model, detect objects and returns a detections.'''
    model = load_yolo_model()
    try:
        logger.info("Starting Object Detection...")
        results = model.predict(image)
        result = results[0]
        detections = create_detections(result)
        if detections:  # if detections were made, list is not empty
            return detections
        else:
            return None
    except Exception as e:
        logger.error(f"Error in detecting objects: {e}")
        raise