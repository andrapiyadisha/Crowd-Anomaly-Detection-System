from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("model/UCF model.pt")

# Video path
video_path = "videos/UCFvideo1.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)

    label = results[0].names[results[0].probs.top1]
    conf = float(results[0].probs.top1conf)

    # Choose color
    if label.lower() == "violence":
        color = (0,0,255)   # red
        warning = "⚠ VIOLENCE DETECTED"
    else:
        color = (0,255,0)   # green
        warning = "NORMAL ACTIVITY"

    text = f"{label} {conf:.2f}"

    # Draw background rectangle
    cv2.rectangle(frame, (10,10), (420,80), (0,0,0), -1)

    # Warning text
    cv2.putText(frame, warning, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2)

    # Prediction text
    cv2.putText(frame, text, (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)

    cv2.imshow("Crowd Anomaly Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()