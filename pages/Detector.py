import cv2
import streamlit as st
from ultralytics import YOLO
import supervision as sv
import time
from openai import OpenAI
import json
import os
import base64
import re
import asyncio
from datetime import datetime
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Load the YOLO model
MODEL_PATH = 'ckpt/yolov8s-worldv2.pt'
if not os.path.exists(MODEL_PATH):
    st.error(f"YOLO model not found at {MODEL_PATH}. Please ensure the model file exists.")
    st.stop()

model_general = YOLO(MODEL_PATH)
model_general.set_classes(["food"])

# Initialize OpenAI client
try:
    openai_api_key = st.secrets["OpenAI_key"]
    client = OpenAI(api_key=openai_api_key)
except KeyError:
    st.error("OpenAI API key not found in secrets.")
    st.stop()

# Directory to save the image and data
IMAGE_CACHE_DIRECTORY = "data"
RESULTS_DB = "food_waste_records.json"
os.makedirs(IMAGE_CACHE_DIRECTORY, exist_ok=True)

# Initialize Supervision Annotators
bbox_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5, text_color=sv.Color.BLACK)

# GPT-4 Classification Function
async def classify_food_with_gpt4(image):
    """
    Use GPT-4 to classify the food waste and ensure the output is in JSON format.
    """
    try:
        # Convert image to base64 string
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Define the prompt with an explicit instruction for JSON output
        prompt = (
            "You are a food waste analyzer. "
            "Classify the food waste in this image and estimate its amount in grams. "
            "Provide the information in the following strict JSON format: "
            "{\"name\": \"...\", \"amount\": ...}. "
            "Ensure the response is in this format without any extra explanation. "
            "The 'name' should be specific enough for later analysis."
            "\nExample output: {\"name\": \"cheese pizza\", \"amount\": 312}"
        )
        
        # Call the OpenAI API
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150
        )
        
        response_text = response.choices[0].message.content.strip()
        st.write(f"Raw GPT-4 response: {response_text}")
        
        # Extract JSON-like content using regular expressions
        json_pattern = r"\{.*\}"
        match = re.search(json_pattern, response_text)
        if match:
            response_text = match.group(0)
        
        # Parse the response as JSON
        try:
            classification = json.loads(response_text)
        except json.JSONDecodeError:
            # If the response is not valid JSON, attempt to fix common issues
            response_text = response_text.replace("'", "\"")  # Replace single quotes with double quotes
            try:
                classification = json.loads(response_text)
            except json.JSONDecodeError:
                st.error("Invalid JSON format received from GPT-4 after correction attempt.")
                return None
        
        return classification
    except Exception as e:
        st.error(f"Error in GPT-4 classification: {e}")
        return None

# Function to save classification results to database
def save_to_database(image_path, timestamp, classification):
    """
    Save the food waste record to a database (JSON file).
    """
    try:
        record = {
            "timestamp": timestamp,
            "image_path": image_path,
            "name": classification['name'],
            "amount": classification['amount']
        }

        if not os.path.exists(RESULTS_DB):
            # Initialize the database file
            with open(RESULTS_DB, 'w') as f:
                json.dump([], f)

        with open(RESULTS_DB, 'r+') as f:
            data = json.load(f)
            data.append(record)
            f.seek(0)
            json.dump(data, f, indent=4)
        
        st.success(f"Saved to database: {record}")
    except Exception as e:
        st.error(f"Error saving to database: {e}")

# Preprocess Image Function
def preprocess_image(frame):
    """
    Preprocess the image for classification.
    """
    try:
        # Example preprocessing: resize the image
        processed_frame = cv2.resize(frame, (224, 224))  # Adjust size if necessary
        return processed_frame
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

# Result Display Function
def show_result(frame, name, amount):
    st.dialog("Detection result")
    FRAME_WINDOW_RESULT = st.image([])
    st.write("### Food detected")
    FRAME_WINDOW_RESULT.image(frame)
    st.write(f"**{name}**, **{amount} g**.")
    if st.button("Confirm"):
        st.experimental_rerun()
        time.sleep(3)

# Video Processor Class for streamlit-webrtc
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detection_count = 0
        self.saved_frame = None
        self.processing = False  # To prevent multiple simultaneous processing

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        annotated_image = img.copy()

        # Perform YOLO detection every frame
        results = model_general(img)
        detections = results[0]

        if detections.boxes:
            # Draw bounding boxes
            boxes = detections.boxes.xyxy.numpy()
            confidences = detections.boxes.conf.numpy()
            class_ids = detections.boxes.cls.numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if conf < 0.5:
                    continue  # Skip low-confidence detections
                label = model_general.names[cls_id]
                cv2.rectangle(annotated_image, (int(box[0]), int(box[1])), 
                              (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(annotated_image, f"{label} {conf:.2f}", 
                            (int(box[0]), int(box[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if label.lower() == "food":
                    self.detection_count += 1
                    if self.detection_count >= 2 and not self.processing:
                        self.processing = True
                        self.saved_frame = img.copy()
                        asyncio.run(self.handle_detection())
                    break
        else:
            self.detection_count = 0  # Reset if no detections

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")
    
    async def handle_detection(self):
        st.toast('Food waste detected! Processing image...')
        time.sleep(1)  # Brief pause

        processed_frame = preprocess_image(self.saved_frame)
        if processed_frame is not None:
            st.toast('Classifying food waste with GPT-4...')
            classification_result = await classify_food_with_gpt4(processed_frame)
            if classification_result is not None:
                food_name = classification_result.get('name')
                amount = classification_result.get('amount')

                st.toast('Recording food waste...')
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                image_filename = f'food_waste_{timestamp}.jpg'
                image_path = os.path.join(IMAGE_CACHE_DIRECTORY, image_filename)
                cv2.imwrite(image_path, self.saved_frame)

                # Save to the database
                save_to_database(image_path, timestamp, classification_result)

                # Show the result to the user
                show_result(cv2.cvtColor(self.saved_frame, cv2.COLOR_BGR2RGB), food_name, amount)
        
        self.processing = False
        self.detection_count = 0  # Reset after processing

# Streamlit UI
st.title("AI Food Waste Tracking with Real-Time Webcam")

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
    key="food-waste-tracking",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Display Saved Records
st.header("Saved Food Waste Records")
if os.path.exists(RESULTS_DB):
    try:
        with open(RESULTS_DB, 'r') as f:
            records = json.load(f)
            if records:
                for record in records:
                    st.subheader(f"{record['timestamp']}")
                    st.image(os.path.join(IMAGE_CACHE_DIRECTORY, record['image_path']), width=300)
                    st.write(f"**Food:** {record['name']}")
                    st.write(f"**Amount:** {record['amount']} g")
            else:
                st.write("No records found.")
    except json.JSONDecodeError:
        st.error("Error reading the database file.")
else:
    st.write("No records found.")

# Additional Information or Controls
st.markdown(
    """
    ### About
    This application uses YOLO for object detection to identify food waste in real-time via your webcam.
    Detected food items are classified using GPT-4, and the results are saved for tracking purposes.

    **Note:** Ensure your webcam is enabled and permissions are granted for this application to function correctly.
    """
)
