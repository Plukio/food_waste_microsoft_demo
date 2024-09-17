import cv2
import streamlit as st
from ultralytics import YOLO
import supervision as sv
import time
import pyautogui
from openai import OpenAI
import json
import os
import base64
import re
import os


# Load the YOLO model
model_general = YOLO('ckpt/yolov8s-worldv2.pt')
model_general.set_classes(["food"])

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") )

@st.dialog("Detection result")
def show_result(frame, name, amount):
    FRAME_WINDOW_RESULT = st.image([])
    st.write(f"Food detected")
    FRAME_WINDOW_RESULT.image(frame)
    st.write(f"{name}, {amount} g.")
    if st.button("Confirm"):
        pyautogui.press('esc')
        st.rerun()
        
        
# Directory to save the image and data
IMAGE_CACHE_DIRECTORY = "data"
RESULTS_DB = "food_waste_records.json"

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

def classify_food_with_gpt4(image):
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
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ],
                }
            ],
            max_tokens=150
        )
        
        response_text = response.choices[0].message.content.strip()
        print(f"Raw response: {response_text}")
        
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

def process_webcam(model):
    bbox = sv.BoxAnnotator(thickness=2)
    food_label = sv.LabelAnnotator(text_thickness=2, text_scale=0.5, text_color=sv.Color.BLACK)

    # Initialize webcam
    camera = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    i = 0
    detection_count = 0  
    saved_frame = None

    try:
        while True:
            # Capture frame from webcam
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            # Detect food every 5th frame
            if i % 10 == 0:
                results_specific = model(frame)
                if len(results_specific[0].boxes.conf) > 0:
                    detection_count += 1
                    if detection_count == 2:
                        st.toast('Food waste detected!')
                        saved_frame = frame
                        break  
                else:
                    detection_count = 0

            # Display the current frame in Streamlit
            show_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(show_frame)
            i += 1
        else:
            st.write('Stopped')

        # Release the webcam
        camera.release()

        # If a frame is saved, process and classify it
        if saved_frame is not None:
            st.toast('Processing image...')
            processed_frame = preprocess_image(saved_frame)
            if processed_frame is not None:
                st.toast('Classifying food waste...')
                classification_result = classify_food_with_gpt4(processed_frame)
                if classification_result is not None:
                    food_name = classification_result.get('name')
                    amount = classification_result.get('amount')

                    st.toast('Recording food waste...')
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    image_path = f'food_waste_{timestamp}.jpg'
                    cv2.imwrite(os.path.join(IMAGE_CACHE_DIRECTORY, image_path), saved_frame)

                    # Save to the database
                    save_to_database(image_path, timestamp, classification_result)

                    st.toast('Finish recording!', icon="📝")
                    show_result(show_frame, food_name, amount)

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        # Ensure the camera is released even if an error occurs
        if camera.isOpened():
            camera.release()

# Streamlit UI
st.title("Webcam Live Feed")

# Run the webcam processing function
if st.button('Start Detection'):
    process_webcam(model_general)
