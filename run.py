# from ultralytics import YOLO

# model = YOLO('runs/detect/people_yolo_gpu1/weights/best.pt')  # Load your trained model

# results = model.predict(source='/home/intozi/Videos/people.mp4', save=True, conf=0.25)


from ultralytics import YOLO

model = YOLO('runs/detect/people_yolo_gpu1/weights/best.pt')  # Load your trained model

results = model.predict(
    source='/home/intozi/Videos/people.mp4', 
    save=True,        # Saves output video to 'runs/detect/predict'
    conf=0.25,        # Confidence threshold
    show=True         # 👈 This opens a window to show video while processing
)
