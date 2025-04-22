# from ultralytics import YOLO

# # Load a pre-trained YOLOv8 model (choose from 'yolov8n', 'yolov8s', etc.)
# model = YOLO('yolov8m.pt')  # or 'yolov8s.pt', 'yolov8m.pt', etc.

# # Train the model
# model.train(
#     data='/home/intozi/Videos/people_detection/people.yaml',
#     epochs=50,
#     imgsz=640,
#     batch=16,         # Optional: change batch size
#     project='runs',   # Optional: where to save results
#     name='people_yolo_training',  # Optional: name of the training folder
#     exist_ok=True     # Overwrite if the folder already exists
# )



import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='/home/intozi/Videos/people_detection/people.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=1,  # Refers to GPU 0 of the visible devices (which is GPU 1 globally)
    name='people_yolo_gpu1',
    exist_ok=True
)
