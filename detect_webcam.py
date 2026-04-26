from ultralytics import YOLO

model = YOLO("model/wepon Detection model.pt")

model(0, show=True)