"""
Configuration file for YOLO Injection Area Segmentation Model
"""
import os

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "train6", "weights", "best.pt")
MODEL_VERSION = "1.0.0"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.7"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

# Class mapping - Based on your trained model
# This maps YOLO model class IDs to Label Studio label names
CLASS_MAPPING = {
    0: "arm_injection_area",  # Your trained model's class
}

# Label Studio configuration
LABEL_STUDIO_TASK_DATA_KEY = "image"  # Key for image data in Label Studio tasks
LABEL_STUDIO_FROM_NAME = "label"      # From name in Label Studio labeling config
LABEL_STUDIO_TO_NAME = "image"        # To name in Label Studio labeling config

# Processing configuration
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "300"))
DEVICE = os.getenv("DEVICE", "auto")  # "auto", "cpu", or "cuda"

# Label Studio connection (for downloading images)
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")
