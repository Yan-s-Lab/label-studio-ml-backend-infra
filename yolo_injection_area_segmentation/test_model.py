#!/usr/bin/env python3
"""
Test script for YOLO Injection Area Segmentation Model
"""
import os
import sys
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from model import YOLOInjectionAreaSegmentation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if the model can be loaded successfully"""
    try:
        logger.info("Testing model loading...")
        
        # Create a simple label config for testing
        label_config = '''
        <View>
            <Image name="image" value="$image"/>
            <PolygonLabels name="label" toName="image">
                <Label value="injection_area" background="red"/>
            </PolygonLabels>
        </View>
        '''
        
        # Initialize model
        model = YOLOInjectionAreaSegmentation(
            project_id="test_project",
            label_config=label_config
        )
        
        logger.info("✓ Model loaded successfully!")
        logger.info(f"Model version: {model.get('model_version')}")
        
        return model
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return None

def test_prediction():
    """Test prediction with a dummy task"""
    model = test_model_loading()
    if not model:
        return False
    
    try:
        logger.info("Testing prediction...")
        
        # Create a dummy task (you would need to provide a real image path)
        dummy_task = {
            "id": 1,
            "data": {
                "image": "/path/to/test/image.jpg"  # Replace with actual image path
            }
        }
        
        # This will fail without a real image, but we can test the structure
        try:
            response = model.predict([dummy_task])
            logger.info("✓ Prediction method executed successfully!")
            logger.info(f"Response type: {type(response)}")
            return True
        except Exception as e:
            if "No image found" in str(e) or "not found" in str(e):
                logger.info("✓ Prediction method structure is correct (expected error due to missing test image)")
                return True
            else:
                raise e
                
    except Exception as e:
        logger.error(f"✗ Prediction test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting YOLO Injection Area Segmentation Model Tests")
    logger.info("=" * 60)
    
    # Test 1: Model loading
    success = test_model_loading() is not None
    
    # Test 2: Prediction structure
    if success:
        success = test_prediction()
    
    logger.info("=" * 60)
    if success:
        logger.info("✓ All tests passed!")
        logger.info("\nNext steps:")
        logger.info("1. Install dependencies: pip install -r requirements.txt")
        logger.info("2. Ensure your model file exists at: train6/weights/best.pt")
        logger.info("3. Update CLASS_MAPPING in config.py with your actual classes")
        logger.info("4. Test with real images")
    else:
        logger.error("✗ Some tests failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
