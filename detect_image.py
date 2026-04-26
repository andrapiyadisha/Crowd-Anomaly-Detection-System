from ultralytics import YOLO

# load model
model = YOLO("model/wepon Detection model.pt")

# run detection
results = model("images/img3.webp", show=True, save=True)

print("Detection completed")