from ultralytics import YOLO

model = YOLO("model/wepon Detection model.pt")

results = model("videos/video6.mp4", show=True, imgsz=640)

