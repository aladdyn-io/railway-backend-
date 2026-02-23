#!/usr/bin/env python3
"""
Test script to verify all 4 detection APIs are working
"""

import requests
import json

def test_video_processor_api():
    """Test the unified video processor API"""
    try:
        # Health check
        response = requests.get('http://localhost:9001/api/health')
        if response.status_code == 200:
            print("✅ Video Processor API (Port 9001) - HEALTHY")
            return True
        else:
            print("❌ Video Processor API (Port 9001) - UNHEALTHY")
            return False
    except Exception as e:
        print(f"❌ Video Processor API (Port 9001) - ERROR: {e}")
        return False

def test_individual_apis():
    """Test individual APIs"""
    apis = [
        ("MediaPipe API", "http://localhost:5001/health"),
        ("YOLO API", "http://localhost:5002/health"), 
        ("OpenCV API", "http://localhost:5003/health")
    ]
    
    results = {}
    for name, url in apis:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name} - HEALTHY")
                results[name] = True
            else:
                print(f"❌ {name} - UNHEALTHY (Status: {response.status_code})")
                results[name] = False
        except Exception as e:
            print(f"❌ {name} - ERROR: {e}")
            results[name] = False
    
    return results

def main():
    print("🔍 Testing Railway Safety Detection APIs...")
    print("=" * 50)
    
    # Test main video processor
    main_api_healthy = test_video_processor_api()
    
    print("\n🔧 Testing Individual APIs:")
    print("-" * 30)
    individual_results = test_individual_apis()
    
    print("\n📊 Detection Capabilities:")
    print("-" * 30)
    print("1. 📱 Phone Detection (YOLO) - Confidence lowered to 25%")
    print("2. 👋 Hand Signal Detection (MediaPipe) - More lenient thresholds") 
    print("3. 😴 Microsleep Detection (Face Analysis) - Eye closure detection")
    print("4. 🎒 Bag Detection (YOLO) - Backpack, handbag, suitcase")
    
    print("\n🎯 Improvements Applied:")
    print("-" * 30)
    print("• Lowered YOLO confidence from 40% → 25%")
    print("• Simplified hand signal detection")
    print("• Process every 15 frames instead of 30")
    print("• Better logging for all detections")
    
    if main_api_healthy:
        print("\n✅ READY TO TEST!")
        print("Upload a video with visible phones, hands, or bags to see all detections working.")
    else:
        print("\n❌ Main API not running. Start with:")
        print("python3 video_processor_api.py")

if __name__ == "__main__":
    main()