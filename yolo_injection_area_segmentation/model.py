import os
import logging
import numpy as np
from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from config import (
    MODEL_PATH, MODEL_VERSION, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    IMAGE_SIZE, CLASS_MAPPING, LABEL_STUDIO_TASK_DATA_KEY,
    LABEL_STUDIO_FROM_NAME, LABEL_STUDIO_TO_NAME, MAX_DETECTIONS, DEVICE
)

try:
    from ultralytics import YOLO
    import torch
    import cv2
    from PIL import Image
except ImportError as e:
    logging.error(f"Required dependencies not installed: {e}")
    raise ImportError("Please install required dependencies: ultralytics, torch, opencv-python, Pillow")

logger = logging.getLogger(__name__)


class YOLOInjectionAreaSegmentation(LabelStudioMLBase):
    """Custom YOLO ML Backend model for injection area segmentation
    """

    def setup(self):
        """Configure any parameters of your model here
        """
        logger.info(f"🔧 Setting up YOLO Injection Area Segmentation model")
        self.set("model_version", MODEL_VERSION)
        self.model = None
        self.load_model()
        logger.info(f"✅ Model setup completed")

    def load_model(self):
        """Load the trained YOLO model"""
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

            logger.info(f"Loading YOLO model from {MODEL_PATH}")
            self.model = YOLO(MODEL_PATH)

            # Set device
            if DEVICE == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = DEVICE

            logger.info(f"Using device: {device}")
            self.model.to(device)

            logger.info(f"Model loaded successfully. Classes: {self.model.names}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
    # 继承了MLBase
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Run YOLO segmentation inference on input images
            :param tasks: Label Studio tasks in JSON format
            :param context: Label Studio context in JSON format
            :return model_response: ModelResponse with predictions
        """
        logger.info(f"🔮 Predict method called with {len(tasks)} tasks")
        logger.debug(f"Tasks received: {tasks}")
        logger.debug(f"Context: {context}")
        logger.debug(f"Kwargs: {kwargs}")

        # Log environment variables for debugging
        logger.debug(f"🌍 Environment variables:")
        logger.debug(f"   - LABEL_STUDIO_URL: {os.getenv('LABEL_STUDIO_URL', 'Not set')}")
        logger.debug(f"   - LABEL_STUDIO_API_KEY: {'Set' if os.getenv('LABEL_STUDIO_API_KEY') else 'Not set'}")

        if not self.model:
            logger.error("❌ Model not loaded")
            return ModelResponse(predictions=[])

        predictions = []

        for i, task in enumerate(tasks):
            logger.info(f"📋 Processing task {i+1}/{len(tasks)}: {task.get('id', 'unknown')}")
            try:
                # Get image from task
                logger.debug(f"Task data: {task.get('data', {})}")
                image_path = self.get_image_path(task)
                if not image_path:
                    logger.warning(f"⚠️ No image found in task {task.get('id', 'unknown')}")
                    logger.debug(f"Task keys: {list(task.keys())}")
                    logger.debug(f"Task data keys: {list(task.get('data', {}).keys())}")
                    # Add empty prediction for tasks without images
                    predictions.append({
                        "model_version": self.get("model_version"),
                        "score": 0.0,
                        "result": []
                    })
                    continue

                logger.info(f"🖼️ Image path: {image_path}")

                # Run YOLO inference
                logger.info(f"🚀 Running YOLO inference with:")
                logger.info(f"   - Confidence threshold: {CONFIDENCE_THRESHOLD}")
                logger.info(f"   - IoU threshold: {IOU_THRESHOLD}")
                logger.info(f"   - Image size: {IMAGE_SIZE}")
                logger.info(f"   - Max detections: {MAX_DETECTIONS}")

                results = self.model.predict(
                    source=image_path,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    imgsz=IMAGE_SIZE,
                    max_det=MAX_DETECTIONS,
                    verbose=False
                )

                logger.info(f"✅ YOLO inference completed. Results count: {len(results) if results else 0}")
                if results and len(results) > 0:
                    result = results[0]
                    logger.info(f"📊 Detection details:")
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        logger.info(f"   - Boxes found: {len(result.boxes)}")
                        if len(result.boxes) > 0:
                            logger.info(f"   - Confidences: {result.boxes.conf.tolist() if result.boxes.conf is not None else 'None'}")
                            logger.info(f"   - Classes: {result.boxes.cls.tolist() if result.boxes.cls is not None else 'None'}")
                    if hasattr(result, 'masks') and result.masks is not None:
                        logger.info(f"   - Masks found: {len(result.masks)}")
                    else:
                        logger.info(f"   - No masks found")

                # Convert results to Label Studio format
                prediction = self.convert_results_to_ls_format(results, task)
                logger.info(f"🔄 Converted prediction: score={prediction.get('score', 0)}, results_count={len(prediction.get('result', []))}")
                logger.debug(f"Full prediction: {prediction}")
                predictions.append(prediction)

            except Exception as e:
                logger.error(f"❌ Error processing task {task.get('id', 'unknown')}: {e}")
                logger.exception("Full traceback:")
                # Add empty prediction for failed tasks
                predictions.append({
                    "model_version": self.get("model_version"),
                    "score": 0.0,
                    "result": []
                })

        logger.info(f"🎯 Prediction completed. Total predictions: {len(predictions)}")
        final_response = ModelResponse(predictions=predictions)
        logger.debug(f"Final response: {final_response}")
        return final_response

    def get_image_path(self, task: Dict) -> Optional[str]:
        """Extract image path from Label Studio task"""
        try:
            logger.debug(f"🔍 Extracting image path from task")
            logger.debug(f"Task data keys: {list(task.get('data', {}).keys())}")
            logger.debug(f"Looking for key: {LABEL_STUDIO_TASK_DATA_KEY}")

            # Try to get image URL from task data
            if LABEL_STUDIO_TASK_DATA_KEY in task.get('data', {}):
                image_url = task['data'][LABEL_STUDIO_TASK_DATA_KEY]
                logger.info(f"📍 Found image URL with key '{LABEL_STUDIO_TASK_DATA_KEY}': {image_url}")

                # Handle different types of image paths
                if image_url.startswith(('http://', 'https://')):
                    # Remote URL - download locally
                    logger.info(f"🌐 Image is a remote URL, downloading locally...")
                    local_path = self.get_local_path(image_url, task_id=task.get('id'))
                    logger.info(f"📥 Downloaded to local path: {local_path}")
                    return local_path
                elif image_url.startswith('/data/local-files/'):
                    # Label Studio local file path - use get_local_path to resolve
                    logger.info(f"🏠 Image is a Label Studio local file, resolving path...")
                    try:
                        local_path = self.get_local_path(image_url, task_id=task.get('id'))
                        logger.info(f"✅ Resolved local path: {local_path}")

                        # Verify the file exists
                        if os.path.exists(local_path):
                            logger.info(f"✅ File exists at: {local_path}")
                            return local_path
                        else:
                            logger.error(f"❌ Resolved file does not exist: {local_path}")
                            return None
                    except Exception as resolve_error:
                        logger.error(f"❌ Error resolving Label Studio path: {resolve_error}")
                        logger.exception("Full traceback for path resolution:")

                        # Fallback: try to manually construct the path
                        logger.info(f"🔄 Attempting fallback path resolution...")
                        fallback_path = self.try_fallback_path_resolution(image_url)
                        if fallback_path:
                            logger.info(f"✅ Fallback resolution successful: {fallback_path}")
                            return fallback_path
                        else:
                            logger.error(f"❌ Fallback resolution failed")
                            return None
                else:
                    # Direct local file path
                    logger.info(f"📁 Image is a direct local path: {image_url}")
                    if os.path.exists(image_url):
                        logger.info(f"✅ File exists at: {image_url}")
                        return image_url
                    else:
                        logger.error(f"❌ File does not exist: {image_url}")
                        return None

            # Fallback: try common image keys
            logger.debug(f"🔄 Primary key not found, trying fallback keys...")
            for key in ['image', 'image_url', 'img', 'photo']:
                if key in task.get('data', {}):
                    image_url = task['data'][key]
                    logger.info(f"📍 Found image URL with fallback key '{key}': {image_url}")

                    if image_url.startswith(('http://', 'https://')):
                        logger.info(f"🌐 Image is a remote URL, downloading locally...")
                        local_path = self.get_local_path(image_url, task_id=task.get('id'))
                        logger.info(f"📥 Downloaded to local path: {local_path}")
                        return local_path
                    elif image_url.startswith('/data/local-files/'):
                        logger.info(f"🏠 Image is a Label Studio local file, resolving path...")
                        try:
                            local_path = self.get_local_path(image_url, task_id=task.get('id'))
                            logger.info(f"✅ Resolved local path: {local_path}")

                            if os.path.exists(local_path):
                                logger.info(f"✅ File exists at: {local_path}")
                                return local_path
                            else:
                                logger.error(f"❌ Resolved file does not exist: {local_path}")
                                return None
                        except Exception as resolve_error:
                            logger.error(f"❌ Error resolving Label Studio path: {resolve_error}")
                            logger.exception("Full traceback for path resolution:")
                            return None
                    else:
                        logger.info(f"📁 Image is a direct local path: {image_url}")
                        if os.path.exists(image_url):
                            logger.info(f"✅ File exists at: {image_url}")
                            return image_url
                        else:
                            logger.error(f"❌ File does not exist: {image_url}")
                            return None

            logger.warning(f"❌ No image found in task data. Available keys: {list(task.get('data', {}).keys())}")
            return None

        except Exception as e:
            logger.error(f"❌ Error getting image path: {e}")
            logger.exception("Full traceback:")
            return None

    def try_fallback_path_resolution(self, image_url: str) -> Optional[str]:
        """Try to resolve Label Studio local file path manually with smart path mapping"""
        try:
            logger.debug(f"🔄 Attempting manual path resolution for: {image_url}")

            # Parse the Label Studio local file URL
            # Format: /data/local-files/?d=synthetic-data/ComfyUI/output/flux_00376_.png
            # Format: /data/local-files/?d=injection-site-real-data/real_data_arm/real105.png
            if '?d=' in image_url:
                # Extract the relative path after ?d=
                relative_path = image_url.split('?d=', 1)[1]
                logger.debug(f"📁 Extracted relative path: {relative_path}")

                # 智能路径映射：根据路径前缀确定正确的基础路径
                resolved_path = self.smart_path_mapping(relative_path)
                if resolved_path:
                    return resolved_path

                # 如果智能映射失败，尝试通用路径
                logger.debug(f"🔄 Smart mapping failed, trying common paths...")
                possible_base_paths = [
                    "/shared-storage",  # 统一的共享存储路径
                    "/home/yan/StudioSpace/AI_Annotation_Studio/core_work_flow/storage",  # 原有路径作为备用
                    "/data",
                    "/app/data",
                    "/opt/heartex/data",
                    os.path.expanduser("~/label-studio-data"),
                    os.path.expanduser("~/.local/share/label-studio/media/upload"),
                ]

                for base_path in possible_base_paths:
                    full_path = os.path.join(base_path, relative_path)
                    logger.debug(f"🔍 Trying path: {full_path}")

                    if os.path.exists(full_path):
                        logger.info(f"✅ Found file at: {full_path}")
                        return full_path

                logger.warning(f"⚠️ File not found in any of the expected locations")
                logger.debug(f"Tried paths: {[os.path.join(bp, relative_path) for bp in possible_base_paths]}")

            else:
                logger.warning(f"⚠️ Unexpected Label Studio URL format: {image_url}")

            return None

        except Exception as e:
            logger.error(f"❌ Error in fallback path resolution: {e}")
            logger.exception("Full traceback:")
            return None

    def smart_path_mapping(self, relative_path: str) -> Optional[str]:
        """智能路径映射：根据相对路径自动确定正确的基础路径"""
        try:
            logger.debug(f"🧠 Smart path mapping for: {relative_path}")

            # 定义路径映射规则
            path_mappings = {
                # 合成数据路径映射
                "synthetic-data/": "/shared-storage/synthetic-data/",
                "ComfyUI/": "/shared-storage/synthetic-data/",  # 简化路径也映射到合成数据

                # 真实数据路径映射
                "injection-site-real-data/": "/shared-storage/injection-site-real-data/",
                "real_data_arm/": "/shared-storage/injection-site-real-data/",  # 简化路径也映射到真实数据
            }

            # 尝试匹配路径前缀
            for prefix, base_path in path_mappings.items():
                if relative_path.startswith(prefix):
                    # 移除前缀，构建完整路径
                    remaining_path = relative_path[len(prefix):]
                    full_path = os.path.join(base_path, remaining_path)
                    logger.debug(f"🎯 Mapped {prefix} -> {base_path}, full path: {full_path}")

                    if os.path.exists(full_path):
                        logger.info(f"✅ Smart mapping successful: {full_path}")
                        return full_path
                    else:
                        logger.debug(f"❌ Mapped path does not exist: {full_path}")

                # 也尝试直接在基础路径下查找
                direct_path = os.path.join(base_path, relative_path)
                if os.path.exists(direct_path):
                    logger.info(f"✅ Direct mapping successful: {direct_path}")
                    return direct_path

            # 如果没有匹配的前缀，尝试在统一的 /shared-storage/ 下查找
            unified_path = os.path.join("/shared-storage", relative_path)
            if os.path.exists(unified_path):
                logger.info(f"✅ Unified path mapping successful: {unified_path}")
                return unified_path

            logger.debug(f"❌ Smart mapping failed for: {relative_path}")
            return None

        except Exception as e:
            logger.error(f"❌ Error in smart path mapping: {e}")
            return None

    def convert_results_to_ls_format(self, results, task: Dict) -> Dict:
        """Convert YOLO results to Label Studio format"""
        try:
            logger.debug(f"🔄 Converting YOLO results to Label Studio format")

            if not results or len(results) == 0:
                logger.info(f"📭 No results to convert")
                return {
                    "model_version": self.get("model_version"),
                    "score": 0.0,
                    "result": []
                }

            result = results[0]  # Take first result
            logger.debug(f"📊 Processing result with attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            predictions = []

            # Check if we have segmentation masks
            if hasattr(result, 'masks') and result.masks is not None:
                logger.info(f"🎭 Found segmentation masks, creating polygon predictions")
                predictions.extend(self.create_polygon_predictions(result))

            # If no masks, fall back to bounding boxes
            elif hasattr(result, 'boxes') and result.boxes is not None:
                logger.info(f"📦 No masks found, creating bounding box predictions")
                predictions.extend(self.create_bbox_predictions(result))
            else:
                logger.warning(f"⚠️ No masks or boxes found in result")

            avg_score = float(np.mean([p.get('score', 0.0) for p in predictions]) if predictions else 0.0)
            logger.info(f"✅ Conversion completed: {len(predictions)} predictions, avg_score={avg_score:.3f}")

            return {
                "model_version": self.get("model_version"),
                "score": avg_score,
                "result": predictions
            }

        except Exception as e:
            logger.error(f"❌ Error converting results: {e}")
            logger.exception("Full traceback:")
            return {
                "model_version": self.get("model_version"),
                "score": 0.0,
                "result": []
            }

    def create_polygon_predictions(self, result) -> List[Dict]:
        """Create polygon predictions from YOLO segmentation masks"""
        predictions = []

        try:
            masks = result.masks
            boxes = result.boxes
            logger.info(f"🎭 Creating polygon predictions from {len(masks)} masks")

            for i in range(len(masks)):
                # Get confidence score
                confidence = float(boxes.conf[i]) if boxes.conf is not None else 0.0
                logger.debug(f"   Mask {i+1}: confidence={confidence:.3f}")

                # Skip low confidence predictions
                if confidence < CONFIDENCE_THRESHOLD:
                    logger.debug(f"   Skipping mask {i+1}: confidence {confidence:.3f} < threshold {CONFIDENCE_THRESHOLD}")
                    continue

                # Get class ID and name
                class_id = int(boxes.cls[i]) if boxes.cls is not None else 0
                class_name = self.get_class_name(class_id)
                logger.debug(f"   Mask {i+1}: class_id={class_id}, class_name={class_name}")

                # Get polygon points (normalized coordinates)
                if hasattr(masks, 'xyn') and masks.xyn is not None:
                    points = masks.xyn[i] * 100  # Convert to percentage
                    points_list = points.tolist()
                    logger.debug(f"   Mask {i+1}: {len(points_list)} polygon points")

                    prediction = {
                        "from_name": LABEL_STUDIO_FROM_NAME,
                        "to_name": LABEL_STUDIO_TO_NAME,
                        "type": "polygonlabels",
                        "value": {
                            "polygonlabels": [class_name],
                            "points": points_list,
                            "closed": True,
                        },
                        "score": confidence,
                    }
                    predictions.append(prediction)
                    logger.info(f"   ✅ Added polygon prediction {len(predictions)}: {class_name} (conf={confidence:.3f})")
                else:
                    logger.warning(f"   ⚠️ No polygon points found for mask {i+1}")

            logger.info(f"🎭 Polygon predictions completed: {len(predictions)} valid predictions")

        except Exception as e:
            logger.error(f"❌ Error creating polygon predictions: {e}")
            logger.exception("Full traceback:")

        return predictions

    def create_bbox_predictions(self, result) -> List[Dict]:
        """Create bounding box predictions from YOLO detection boxes"""
        predictions = []

        try:
            boxes = result.boxes
            logger.info(f"📦 Creating bbox predictions from {len(boxes)} boxes")

            for i in range(len(boxes)):
                # Get confidence score
                confidence = float(boxes.conf[i]) if boxes.conf is not None else 0.0
                logger.debug(f"   Box {i+1}: confidence={confidence:.3f}")

                # Skip low confidence predictions
                if confidence < CONFIDENCE_THRESHOLD:
                    logger.debug(f"   Skipping box {i+1}: confidence {confidence:.3f} < threshold {CONFIDENCE_THRESHOLD}")
                    continue

                # Get class ID and name
                class_id = int(boxes.cls[i]) if boxes.cls is not None else 0
                class_name = self.get_class_name(class_id)
                logger.debug(f"   Box {i+1}: class_id={class_id}, class_name={class_name}")

                # Get bounding box coordinates (normalized)
                if hasattr(boxes, 'xywhn') and boxes.xywhn is not None:
                    x_center, y_center, width, height = boxes.xywhn[i]

                    # Convert to Label Studio format (percentage)
                    x = float((x_center - width/2) * 100)
                    y = float((y_center - height/2) * 100)
                    w = float(width * 100)
                    h = float(height * 100)

                    logger.debug(f"   Box {i+1}: x={x:.1f}%, y={y:.1f}%, w={w:.1f}%, h={h:.1f}%")

                    prediction = {
                        "from_name": LABEL_STUDIO_FROM_NAME,
                        "to_name": LABEL_STUDIO_TO_NAME,
                        "type": "rectanglelabels",
                        "value": {
                            "rectanglelabels": [class_name],
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                        },
                        "score": confidence,
                    }
                    predictions.append(prediction)
                    logger.info(f"   ✅ Added bbox prediction {len(predictions)}: {class_name} (conf={confidence:.3f})")
                else:
                    logger.warning(f"   ⚠️ No bbox coordinates found for box {i+1}")

            logger.info(f"📦 Bbox predictions completed: {len(predictions)} valid predictions")

        except Exception as e:
            logger.error(f"❌ Error creating bbox predictions: {e}")
            logger.exception("Full traceback:")

        return predictions

    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        # First try the model's built-in names
        if hasattr(self.model, 'names') and class_id in self.model.names:
            model_class_name = self.model.names[class_id]

            # Check if we have a mapping for this class
            if class_id in CLASS_MAPPING:
                return CLASS_MAPPING[class_id]
            elif model_class_name in CLASS_MAPPING.values():
                return model_class_name
            else:
                return model_class_name

        # Fallback to class mapping
        return CLASS_MAPPING.get(class_id, f"class_{class_id}")
    
    # 继承了MLBase
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """
        logger.info(f"Fit method called with event: {event}")

        # For now, we'll just log the event and data
        # In a production setup, you might want to:
        # 1. Collect new annotations
        # 2. Retrain the model periodically
        # 3. Update model weights

        # Store event information
        self.set('last_training_event', event)
        self.set('last_training_data', str(data)[:1000])  # Truncate for storage

        logger.info('Fit method completed successfully.')


# Create an alias for backward compatibility
NewModel = YOLOInjectionAreaSegmentation

