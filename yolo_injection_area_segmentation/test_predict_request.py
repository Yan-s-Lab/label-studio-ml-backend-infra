#!/usr/bin/env python3
"""
Test script to simulate Label Studio prediction requests
"""

import requests
import json
import os

# Test configuration
ML_BACKEND_URL = "http://localhost:9090"
TEST_IMAGE_PATH = "/home/yan/StudioSpace/AI_Annotation_Studio/core_work_flow/storage/ComfyUI/output/flux_00376_.png"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{ML_BACKEND_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_predict_with_local_path():
    """Test prediction with local file path"""
    print("\n🔮 Testing prediction with local file path...")
    
    # Check if test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Test image not found: {TEST_IMAGE_PATH}")
        return False
    
    # Prepare request payload
    request_data = {
        "tasks": [{
            "id": 999,
            "data": {
                "image": TEST_IMAGE_PATH
            },
            "meta": {},
            "created_at": "2025-05-22T22:00:00.000000Z",
            "updated_at": "2025-05-22T22:00:00.000000Z",
            "is_labeled": False,
            "overlap": 1,
            "inner_id": 999,
            "total_annotations": 0,
            "cancelled_annotations": 0,
            "total_predictions": 0,
            "comment_count": 0,
            "unresolved_comment_count": 0,
            "last_comment_updated_at": None,
            "project": 3,
            "updated_by": None,
            "file_upload": None,
            "comment_authors": [],
            "annotations": [],
            "predictions": []
        }],
        "project": "3.1747795083",
        "label_config": """<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="tag" toName="image">
    <Label value="injection_area_arm" background="#FFA39E"/>
  </BrushLabels>
</View>""",
        "params": {
            "login": None,
            "password": None,
            "context": None
        }
    }
    
    try:
        print(f"📤 Sending request to {ML_BACKEND_URL}/predict")
        print(f"📁 Test image: {TEST_IMAGE_PATH}")
        
        response = requests.post(
            f"{ML_BACKEND_URL}/predict",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"📊 Results: {json.dumps(result, indent=2)}")
            
            # Check if we got predictions
            results = result.get('results', [])
            if results and len(results) > 0:
                first_result = results[0]
                predictions = first_result.get('result', [])
                score = first_result.get('score', 0)
                
                print(f"\n📈 Analysis:")
                print(f"   - Number of predictions: {len(predictions)}")
                print(f"   - Average score: {score}")
                print(f"   - Model version: {first_result.get('model_version', 'Unknown')}")
                
                if predictions:
                    print(f"   - Prediction types: {[p.get('type', 'unknown') for p in predictions]}")
                    return True
                else:
                    print(f"   ⚠️ No predictions found (empty result)")
                    return False
            else:
                print(f"❌ No results in response")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_predict_with_label_studio_path():
    """Test prediction with Label Studio local file path format"""
    print("\n🏠 Testing prediction with Label Studio local file path...")

    # Prepare request payload with Label Studio format path
    request_data = {
        "tasks": [{
            "id": 998,
            "data": {
                "image": "/data/local-files/?d=synthetic-data/ComfyUI/output/flux_00177_.png"
            },
            "meta": {},
            "created_at": "2025-05-22T22:00:00.000000Z",
            "updated_at": "2025-05-22T22:00:00.000000Z",
            "is_labeled": False,
            "overlap": 1,
            "inner_id": 998,
            "total_annotations": 0,
            "cancelled_annotations": 0,
            "total_predictions": 0,
            "comment_count": 0,
            "unresolved_comment_count": 0,
            "last_comment_updated_at": None,
            "project": 3,
            "updated_by": None,
            "file_upload": None,
            "comment_authors": [],
            "annotations": [],
            "predictions": []
        }],
        "project": "3.1747795083",
        "label_config": """<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="tag" toName="image">
    <Label value="injection_area_arm" background="#FFA39E"/>
  </BrushLabels>
</View>""",
        "params": {
            "login": None,
            "password": None,
            "context": None
        }
    }
    
    try:
        print(f"📤 Sending request to {ML_BACKEND_URL}/predict")
        print(f"🏠 Label Studio path: /data/local-files/?d=synthetic-data/ComfyUI/output/flux_00177_.png")
        
        response = requests.post(
            f"{ML_BACKEND_URL}/predict",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful!")
            print(f"📊 Results: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_multiple_label_studio_paths():
    """Test multiple Label Studio path formats to diagnose the issue"""
    print("\n🔍 Testing multiple Label Studio path formats...")

    test_paths = [
        # 完整路径格式
        "/data/local-files/?d=synthetic-data/ComfyUI/output/flux_00177_.png",
        "/data/local-files/?d=injection-site-real-data/real_data_arm/real105.png",
        "/data/local-files/?d=synthetic-data/ComfyUI/output/flux_00376_.png",

        # 简化路径格式（测试智能映射）
        "/data/local-files/?d=ComfyUI/output/flux_00177_.png",
        "/data/local-files/?d=real_data_arm/real105.png",
        "/data/local-files/?d=ComfyUI/output/flux_00376_.png"
    ]

    results = {}

    for i, test_path in enumerate(test_paths):
        print(f"\n📍 Test {i+1}: {test_path}")

        request_data = {
            "tasks": [{
                "id": 900 + i,
                "data": {"image": test_path},
                "meta": {},
                "created_at": "2025-05-22T22:00:00.000000Z",
                "updated_at": "2025-05-22T22:00:00.000000Z",
                "is_labeled": False,
                "overlap": 1,
                "inner_id": 900 + i,
                "total_annotations": 0,
                "cancelled_annotations": 0,
                "total_predictions": 0,
                "comment_count": 0,
                "unresolved_comment_count": 0,
                "last_comment_updated_at": None,
                "project": 3,
                "updated_by": None,
                "file_upload": None,
                "comment_authors": [],
                "annotations": [],
                "predictions": []
            }],
            "project": "3.1747795083",
            "label_config": """<View>
  <Image name="image" value="$image" zoom="true"/>
  <BrushLabels name="tag" toName="image">
    <Label value="injection_area_arm" background="#FFA39E"/>
  </BrushLabels>
</View>""",
            "params": {
                "login": None,
                "password": None,
                "context": None
            }
        }

        try:
            response = requests.post(
                f"{ML_BACKEND_URL}/predict",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                print(f"   ✅ Success (200)")
                results[test_path] = "✅ Success"
            else:
                print(f"   ❌ Failed ({response.status_code})")
                results[test_path] = f"❌ Failed ({response.status_code})"

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[test_path] = f"❌ Error: {e}"

    print(f"\n📊 Path Test Results:")
    for path, result in results.items():
        print(f"   {result}: {path}")

    return results

def main():
    """Run all tests"""
    print("🧪 Starting ML Backend Tests")
    print("=" * 50)

    # Test health
    health_ok = test_health()
    if not health_ok:
        print("❌ Health check failed, stopping tests")
        return

    # Test with local path
    local_ok = test_predict_with_local_path()

    # Test with Label Studio path
    ls_ok = test_predict_with_label_studio_path()

    # Test multiple paths to diagnose issues
    path_results = test_multiple_label_studio_paths()

    print("\n" + "=" * 50)
    print("🏁 Test Summary:")
    print(f"   Health check: {'✅' if health_ok else '❌'}")
    print(f"   Local path prediction: {'✅' if local_ok else '❌'}")
    print(f"   Label Studio path prediction: {'✅' if ls_ok else '❌'}")
    print(f"   Path diagnosis completed: {len(path_results)} paths tested")

if __name__ == "__main__":
    main()
