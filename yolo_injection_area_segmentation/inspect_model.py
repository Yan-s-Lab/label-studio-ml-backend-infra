#!/usr/bin/env python3
"""
Utility script to inspect the trained YOLO model and extract class information
"""
import os
import sys

def inspect_model():
    """Inspect the trained YOLO model"""
    try:
        from ultralytics import YOLO
        import torch
        
        model_path = os.path.join(os.path.dirname(__file__), "train6", "weights", "best.pt")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at: {model_path}")
            print("Please ensure your trained model exists at train6/weights/best.pt")
            return False
        
        print(f"🔍 Inspecting model: {model_path}")
        print("=" * 60)
        
        # Load model
        model = YOLO(model_path)
        
        # Basic model info
        print(f"📊 Model Architecture: {model.model.__class__.__name__}")
        print(f"📏 Model Size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        
        # Class information
        print(f"\n🏷️  Model Classes ({len(model.names)} total):")
        print("-" * 40)
        for class_id, class_name in model.names.items():
            print(f"  {class_id}: {class_name}")
        
        # Generate config.py class mapping
        print(f"\n⚙️  Suggested CLASS_MAPPING for config.py:")
        print("-" * 40)
        print("CLASS_MAPPING = {")
        for class_id, class_name in model.names.items():
            print(f"    {class_id}: \"{class_name}\",")
        print("}")
        
        # Model task type
        task = getattr(model, 'task', 'unknown')
        print(f"\n🎯 Task Type: {task}")
        
        # Device info
        device = next(model.model.parameters()).device
        print(f"💻 Current Device: {device}")
        
        # Try a dummy prediction to test model
        print(f"\n🧪 Testing model inference...")
        try:
            # Create a dummy image tensor
            dummy_input = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                results = model.predict(dummy_input, verbose=False)
            print("✅ Model inference test successful!")
            
            # Check if model supports segmentation
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                print("🎭 Segmentation masks: Supported")
            else:
                print("📦 Segmentation masks: Not available (will use bounding boxes)")
                
        except Exception as e:
            print(f"⚠️  Model inference test failed: {e}")
        
        print("\n" + "=" * 60)
        print("✅ Model inspection completed!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Required dependencies not installed: {e}")
        print("Please install: pip install ultralytics torch")
        return False
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")
        return False

def main():
    """Main function"""
    print("🔍 YOLO Model Inspector")
    print("=" * 60)
    
    success = inspect_model()
    
    if success:
        print("\n📝 Next Steps:")
        print("1. Update CLASS_MAPPING in config.py with the suggested mapping above")
        print("2. Adjust CONFIDENCE_THRESHOLD if needed")
        print("3. Test the model with: python test_model.py")
        print("4. Start the ML backend: label-studio-ml start ./")
    else:
        print("\n❌ Model inspection failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
