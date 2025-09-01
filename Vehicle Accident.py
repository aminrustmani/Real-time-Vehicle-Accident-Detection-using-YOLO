import cv2
from ultralytics import YOLO

# Load model with "vehicle" and "accident" classes
model = YOLO("vehiclee.pt")   # or "vehicle.pt"

cap = cv2.VideoCapture("111.MOV")

# Create full screen window
cv2.namedWindow("Accident Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Accident Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.10, iou=0.1)
    accident_detected = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            conf = float(box.conf)

            # --- Filter accidents separately ---
            if cls_name == "accident":
                if conf > 0.6:   # require higher confidence for accidents
                    accident_detected = True
                else:
                    continue  # skip weak accident predictions

    if accident_detected:
        print("⚠️ Accident Detected!")

    # Draw YOLO annotations
    annotated = results[0].plot()

    # Show directly in fullscreen
    cv2.imshow("Accident Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
