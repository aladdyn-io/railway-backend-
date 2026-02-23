import requests
import cv2
import base64
import json

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"

def test_api(endpoint, image_data):
    """Test API endpoint"""
    url = f"http://localhost:5000/api/{endpoint}"
    payload = {"image": image_data}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Create a test image (you can replace with actual image path)
    cap = cv2.VideoCapture(0)  # Use webcam
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite("test_frame.jpg", frame)
        image_data = encode_image("test_frame.jpg")
        
        # Test all endpoints
        endpoints = ["hand-signal", "mobile-phone", "microsleep", "packing"]
        
        for endpoint in endpoints:
            print(f"\nTesting {endpoint}:")
            result = test_api(endpoint, image_data)
            print(json.dumps(result, indent=2))
    else:
        print("Could not capture frame from webcam")