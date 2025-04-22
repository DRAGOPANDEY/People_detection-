# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import os
# import tempfile
# from PIL import Image

# # Load your trained model
# model = YOLO('runs/detect/people_yolo_gpu1/weights/best.pt')

# st.title("👀 People Detection using YOLOv8")
# st.write("Upload an image or video to detect people.")

# option = st.radio("Select Media Type:", ["Image", "Video"])

# if option == "Image":
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         img = Image.open(uploaded_file)
#         st.image(img, caption="Uploaded Image", use_column_width=True)
        
#         # Save to temp file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
#             img.save(temp.name)
#             results = model.predict(source=temp.name, save=True, conf=0.3)
            
#             # Load the result image
#             result_path = results[0].save_dir + '/' + os.path.basename(temp.name)
#             st.image(result_path, caption="Detected Image", use_column_width=True)

# elif option == "Video":
#     uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
    
#     if uploaded_video is not None:
#         # Save video to temp file
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         tfile.write(uploaded_video.read())

#         st.video(tfile.name)  # Show original video

#         st.info("Processing video... please wait ⏳")

#         results = model.predict(source=tfile.name, save=True, conf=0.3)
#         output_path = results[0].save_dir + '/' + os.path.basename(tfile.name)

#         # Display result
#         st.video(output_path)



# import streamlit as st
# from ultralytics import YOLO
# import tempfile
# import os
# import cv2
# import time
# from PIL import Image

# # Load model
# model = YOLO('runs/detect/people_yolo_gpu1/weights/best.pt')

# st.set_page_config(page_title="YOLOv8 People Detection", layout="centered")
# st.title("👀 Real-Time People Detection using YOLOv8")
# st.markdown("Upload an **image** or **video**, and watch YOLOv8 detect people in action!")

# media_type = st.radio("Choose Media Type:", ["Image", "Video"], horizontal=True)

# if media_type == "Image":
#     image_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])
#     if image_file:
#         image = Image.open(image_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
#             image.save(temp.name)
#             with st.spinner("Running YOLOv8 detection..."):
#                 results = model.predict(source=temp.name, conf=0.3)
#                 detected_img_path = results[0].save_dir + '/' + os.path.basename(temp.name)
#                 st.success("Detection complete!")
#                 st.image(detected_img_path, caption="Detected Image", use_column_width=True)

# elif media_type == "Video":
#     video_file = st.file_uploader("🎥 Upload a Video", type=["mp4", "avi", "mov"])
#     if video_file:
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#         tfile.write(video_file.read())
#         st.video(tfile.name)

#         st.info("Processing video and displaying with detections...")

#         # Open video
#         cap = cv2.VideoCapture(tfile.name)
#         stframe = st.empty()

#         # Process frame by frame
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 break

#             # Run inference on the frame
#             results = model.predict(source=frame, conf=0.3, save=False, verbose=False)
#             annotated_frame = results[0].plot()  # Draw detections on frame

#             # Convert BGR to RGB
#             annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#             stframe.image(annotated_frame, channels="RGB", use_column_width=True)
#             time.sleep(0.03)  # ~30 FPS

#         cap.release()
#         st.success("Video processing complete 🎉")




import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os
import time
from collections import defaultdict

# Load YOLO model
model = YOLO('runs/detect/people_yolo_gpu1/weights/best.pt')

st.set_page_config(page_title="YOLOv8 People Detection", layout="centered")
st.title("🧍 People Detection with Tracking & Confidence")
st.markdown("Upload a **video** to detect people, count them, display confidence, and assign IDs.")

# File uploader
video_file = st.file_uploader("🎥 Upload a Video", type=["mp4", "avi", "mov"])

if video_file:
    # Save video to temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())

    # Display placeholder for the processed video
    stframe = st.empty()

    # Create directory to save detection frames
    frame_save_dir = "saved_frames"
    os.makedirs(frame_save_dir, exist_ok=True)

    # Read video
    cap = cv2.VideoCapture(tfile.name)
    frame_count = 0
    unique_ids = set()
    confidence_display = True  # toggle confidence display
    font = cv2.FONT_HERSHEY_SIMPLEX

    with st.spinner("Detecting people..."):
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run YOLOv8 inference
            results = model.track(source=frame, persist=True, stream=False, conf=0.3, verbose=False)

            # Annotate results
            for r in results:
                frame = r.plot(labels=True, conf=confidence_display)

                if hasattr(r, 'boxes') and r.boxes.id is not None:
                    ids = r.boxes.id.cpu().numpy().astype(int)
                    unique_ids.update(ids)

            # Save the frame
            save_path = os.path.join(frame_save_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(save_path, frame)

            # Display annotated frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

            frame_count += 1
            time.sleep(0.03)

        cap.release()

    st.success(f"✅ Detection complete. Total Unique People Detected: **{len(unique_ids)}**")
    st.markdown(f"📁 Saved annotated frames in: `{frame_save_dir}`")
