import cv2
import torch
import numpy as np

# Load YOLOv7 model using PyTorch hub
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt', trust_repo=True)
model.eval()

# Car-related class names in COCO
car_labels = ['car', 'bus', 'truck', 'motorbike']

# Open video
cap = cv2.VideoCapture("intersection.mp4")

# Multi-object tracker dictionary
trackers = []
colors = []

frame_count = 0
detect_interval = 30  # Run detection every 30 frames

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Detection every N frames
    if frame_count % detect_interval == 1:
        trackers = []
        colors = []

        # Run YOLOv7 detection
        results = model(frame)
        detections = results.xyxy[0]  # [x1, y1, x2, y2, confidence, class]

        for *box, conf, cls in detections:
            label = model.names[int(cls)]
            if label in car_labels:
                x1, y1, x2, y2 = map(int, box)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                trackers.append(tracker)
                colors.append(np.random.randint(0, 255, size=3).tolist())

    # Update all trackers
    for i, tracker in enumerate(trackers):
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = map(int, box)
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'Car {i + 1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLOv7 Car Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
