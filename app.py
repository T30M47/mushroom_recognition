import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import gdown
import os

# Define the Google Drive file IDs for the models
TFLITE_MODEL_URL = 'https://drive.google.com/uc?id=1vyAleBTqYR6sXRUxV_2q7IgeHAwf2p4h'  # Replace with your actual file ID for TFLite model


if not os.path.exists('inception_model.tflite'):
    gdown.download(TFLITE_MODEL_URL, 'inception_model.tflite', quiet=False)

# Load YOLO model
from ultralytics import YOLO
yolo_model = YOLO('best.pt')

# Load TFLite model for InceptionResNetV2 classification
interpreter = tf.lite.Interpreter(model_path='inception_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class names for classification
CLASS_NAMES = ['Koprenka', 'Krasnica', 'Muhara', 'Pecurka', 'Rudoliska', 'Rujnica', 'Slinavka', 'Vlaznica', 'Vrganj']

# Detection and Classification Function
def detect_and_classify(image):
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # YOLO Detection
    results = yolo_model(image_cv)
    detections = results[0].boxes.data  # Bounding box data

    # Classify the entire image once (instead of each box)
    image_resized = cv2.resize(image_cv, (224, 224))
    image_array = image_resized / 255.0
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

    # TFLite Classification
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class for the entire image
    predicted_class = CLASS_NAMES[np.argmax(output_data)]

    # Now draw the bounding boxes and predictions with the same class
    y_offset = 30  # To avoid overlapping texts
    for box in detections:
        x1, y1, x2, y2, conf, cls = map(int, box[:6])
        # Draw Rectangle
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate text size and adjust position to avoid overflow
        text_size = cv2.getTextSize(predicted_class, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_width, text_height = text_size

        # Adjust the y position to prevent overlapping with the previous text
        text_y = y1 - 10
        if text_y - text_height < 0:  # If text goes above the image, move it down
            text_y = y1 + text_height + 10

        # Draw the predicted class label above the bounding box
        cv2.putText(image_cv, predicted_class, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    result_img = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_img)

# Gradio Interface
iface = gr.Interface(
    fn=detect_and_classify,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Mushroom Detection & Classification",
    description="Upload an image to detect mushrooms using YOLOv8 and classify them with a TFLite model."
)

# Launch Gradio app with public URL
iface.launch(share=True)
