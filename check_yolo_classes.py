from ultralytics import YOLO

# Load YOLO model
yolo = YOLO('yolov8s.pt')

# Print all class names and IDs
print("YOLO Class IDs and Names:")
print("=" * 40)
for class_id, class_name in yolo.names.items():
    if 'phone' in class_name.lower() or 'cell' in class_name.lower() or 'bag' in class_name.lower() or 'backpack' in class_name.lower() or 'handbag' in class_name.lower() or 'suitcase' in class_name.lower():
        print(f"ID {class_id}: {class_name}")

print("\nAll classes:")
for class_id, class_name in yolo.names.items():
    print(f"{class_id}: {class_name}")