import cv2
from ultralytics import YOLO
model=YOLO(r"best.pt")

class_names = model.names  # List of class names
image_path=r"Dataset\test\images\Accidents-online-video-cutter_com-_mp4-74_jpg.rf.c9a73363436bfd4eec233af8c9839319.jpg"
# # Load the image using OpenCV
image = cv2.imread(image_path)

results = model(image)

# Iterate through results and draw boxes with labels
for result in results:
    for box in result.boxes:
        # Get the coordinates, confidence, and class label
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
        conf = box.conf.item()  # Confidence score
        class_id = int(box.cls.item())  # Class ID

        # Prepare the label text
        label = f"{class_names[class_id]}: {conf:.2f}"

        # Draw the bounding box (blue color, thickness of 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw the label above the bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_size, _ = cv2.getTextSize(label, font, 0.5, 1)
        label_ymin = max(y1, label_size[1] + 10)
        cv2.rectangle(image, (x1, label_ymin - label_size[1] - 10), 
                      (x1 + label_size[0], label_ymin + 4), (255, 0, 0), -1)  # Draw label background
        cv2.putText(image, label, (x1, label_ymin), font, 0.5, (255, 255, 255), 1)  # Put label text

# Save or display the image
cv2.imwrite('path_to_save_output_image.jpg', image)
