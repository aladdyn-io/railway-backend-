#!/usr/bin/env python3
"""
Test script for all 3 APIs
"""
import requests
import json

def test_api(api_name, base_url):
    print(f"\n{'='*50}")
    print(f"Testing {api_name}")
    print(f"{'='*50}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health check: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ Root endpoint: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    return True

# Test all APIs
apis = {
    "MediaPipe API": "http://localhost:5001",
    "YOLO API": "http://localhost:5002",
    "OpenCV API": "http://localhost:5003"
}

print("🧪 Testing All APIs...")
for name, url in apis.items():
    test_api(name, url)

print("\n" + "="*50)
print("✅ API Testing Complete!")
print("="*50)
print("\nTo test with actual images/videos:")
print("  curl -X POST http://localhost:5001/detect/hands -F 'image=@your_image.jpg'")
print("  curl -X POST http://localhost:5002/detect/phone -F 'image=@your_image.jpg'")
print("  curl -X POST http://localhost:5003/video/info -F 'video=@your_video.mp4'")
