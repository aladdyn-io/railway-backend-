#!/usr/bin/env python3
"""
Test phone and bag detection with sample image
"""

import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
yolo = YOLO('yolov8s.pt')

# Create a test image (you can replace this with actual image path)
def test_detection():
    print("🔍 Testing Phone and Bag Detection...")
    print("=" * 50)
    
    # Test with very low confidence
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image for testing
    
    # Run detection with low confidence
    results = yolo.predict(test_image, verbose=False, conf=0.15)[0]
    
    print(f"📊 Detection Results (conf >= 0.15):")
    print(f"Total detections: {len(results.boxes) if results.boxes is not None else 0}")
    
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = yolo.names[cls_id]
            
            if cls_id in [67, 24, 26, 28, 63]:  # phone, bags, laptop
                print(f"✅ {class_name} (ID: {cls_id}) - Confidence: {conf:.3f}")
    
    print("\n🎯 Target Classes:")
    print("📱 Phone: ID 67 (cell phone)")
    print("🎒 Bags: ID 24 (backpack), ID 26 (handbag), ID 28 (suitcase)")
    print("💻 Laptop: ID 63 (laptop)")
    
    print("\n✅ API Updated with:")
    print("• Confidence lowered to 15% (was 25%)")
    print("• Fixed bag class IDs: 24, 26, 28 (was 24, 25, 26)")
    print("• Enhanced logging for all detections")
    print("• Added laptop detection as phone substitute")
    
    print("\n🚀 Ready to test with real video!")
    print("Upload a video with phones or bags visible.")

if __name__ == "__main__":
    test_detection()