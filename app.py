import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import gdown
import os
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic

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

# Define class names and categories
CLASS_NAMES = ['Koprenka', 'Krasnica', 'Muhara', 'Pecurka', 'Rudoliska', 'Rujnica', 'Slinavka', 'Vlaznica', 'Vrganj']
CATEGORIES = {
    "otrovne": ['Koprenka', 'Krasnica', 'Muhara', 'Rudoliska'],
    "jestive": ['Pecurka', 'Rujnica', 'Slinavka', 'Vrganj'],
    "zaštićene": ['Vlaznica']
}
CATEGORY_COLORS = {
    "otrovne": (0, 0, 255),      # Red
    "jestive": (0, 255, 0),      # Green
    "zaštićene": (255, 0, 0)   # Blue
}

# Wrapper for TFLite model to work with LIME
class TFLiteModelWrapper:
    def __init__(self, interpreter, input_details, output_details):
        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details

    def predict_proba(self, images):
        outputs = []
        for image in images:
            # Resize and normalize the image
            image_resized = cv2.resize(image, (224, 224)) / 255.0
            image_array = np.expand_dims(image_resized, axis=0).astype(np.float32)

            # Predict using TFLite model
            self.interpreter.set_tensor(self.input_details[0]['index'], image_array)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            outputs.append(output_data[0])
        return np.array(outputs)

# Initialize LIME explainer
explainer = lime_image.LimeImageExplainer()
model_wrapper = TFLiteModelWrapper(interpreter, input_details, output_details)

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
    confidence_score = np.max(output_data)  # Get the confidence score

    # Prepare confidence scores for all classes
    confidence_text = "Pouzdanost modela po klasama (%):\n"
    for i, score in enumerate(output_data[0]):
        confidence_text += f"{CLASS_NAMES[i]}: {score * 100:.2f}%\n"

    # Determine category and color
    category = next((cat for cat, names in CATEGORIES.items() if predicted_class in names), None)
    color = CATEGORY_COLORS.get(category, (255, 255, 255))  # Default to white if category not found

    # Draw the bounding boxes and predictions with the appropriate color
    for box in detections:
        x1, y1, x2, y2, conf, cls = map(int, box[:6])
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)

        # Prepare text with predicted class and confidence score
        label_text = f"{predicted_class} ({confidence_score*100:.2f}%)"

        # Calculate text size and adjust position to avoid overflow
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_width, text_height = text_size

        # Adjust the y position to prevent overlapping with the previous text
        text_y = y1 - 10
        if text_y - text_height < 0:  # If text goes above the image, move it down
            text_y = y1 + text_height + 10

        # Draw the predicted class label with confidence score above the bounding box
        cv2.putText(image_cv, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Return the annotated image and confidence text (without Grad-CAM overlay)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_cv), confidence_text

def custom_segmentation(image):
     return slic(image, n_segments=50, compactness=10, sigma=1)

def lime_explanation(image):
    original_size = image.size
    small_image = cv2.resize(np.array(image), (224, 224))

    explanation = explainer.explain_instance(
        small_image,
        model_wrapper.predict_proba,
        top_labels=1,
        hide_color=0,
        num_samples=400,
        segmentation_fn=custom_segmentation
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    lime_image = mark_boundaries(temp, mask)
    lime_resized = cv2.resize((lime_image * 255).astype(np.uint8), original_size)

    return Image.fromarray(lime_resized)


# Generate a legend image
def generate_legend():
    legend = np.zeros((150, 400, 3), dtype=np.uint8) + 255

    colors = [("Otrovne", CATEGORY_COLORS["otrovne"]),
              ("Jestive", CATEGORY_COLORS["jestive"]),
              ("Zaštićene", CATEGORY_COLORS["zaštićene"])]

    for i, (label, color) in enumerate(colors):
        y_start = 10 + i * 40
        y_end = y_start + 30

        # Draw color rectangle
        cv2.rectangle(legend, (10, y_start), (40, y_end), color, -1)

        # Put label text
        cv2.putText(legend, label, (50, y_end - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    return Image.fromarray(legend)

def run_detection(image):
    return detect_and_classify(image)


def run_lime(image):
    return lime_explanation(image)

def clear_inputs():
    return None, None, None, None, gr.update(visible=False)

# Define Gradio Interface
with gr.Blocks() as iface:
    gr.Markdown(
        """
        <h1 style='text-align: center;'>Sustav za prepoznavanje gljiva</h1>
        <br><br>
        <h1 style='color:red; font-size: 21px'>!!! UPOZORENJE !!!</h1>
        <p style='font-size: 18px;'>Ova aplikacija koristi model umjetne inteligencije za prepoznavanje gljiva i time može pogriješiti.
        Preporučujemo da ne konzumirate gljive na temelju rezultata ove aplikacije bez dodatne provjere.</p>
        <br>
        <p style='font-size: 18px;'>Model može prepoznati 9 različitih vrsta gljiva navedenih u legendi, a uz predikciju navedena je i pouzdanost modela.
        Nakon predikcije, generiraju se postoci sigurnosti modela po svim klasama te gumb za generiranje LIME objašnjenja.</p>
        <p style='font-size: 18px;'>Na LIME objašnjenje se treba čekati nešto duže vrijeme (~100s), a ono prikazuje dijelove slike koji su najviše utjecali na predikciju modela.</p>
        <div style='margin-top: 20px; font-size: 16px;'>
        <h3 style='font-size: 20px;'>Legenda:</h3>
        <div style='display: flex; align-items: center; margin-bottom: 5px;'>
            <div style='width: 25px; height: 25px; background-color: red; margin-right: 10px;'></div>
            <span>Otrovne gljive - Koprenka, Krasnica, Muhara, Rudoliska</span>
        </div>
        <div style='display: flex; align-items: center; margin-bottom: 5px;'>
            <div style='width: 25px; height: 25px; background-color: green; margin-right: 10px;'></div>
            <span>Jestive gljive - Pečurka, Rujnica, Slinavka, Vrganj</span>
        </div>
        <div style='display: flex; align-items: center;'>
            <div style='width: 25px; height: 25px; background-color: blue; margin-right: 10px;'></div>
            <span>Zaštićene gljive - Vlažnica</span>
        </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Prenesite sliku")

            with gr.Row():
                predict_button = gr.Button("Pokreni detekciju")
                lime_button = gr.Button("Generiraj LIME objašnjenje", visible=False)
            with gr.Row():
                clear_button = gr.Button("Očisti unos")  # Clear button

        with gr.Column():
            pred_image = gr.Image(type="pil", label="Prepoznata gljiva")
            confidence_text = gr.Textbox(label="Postoci pouzdanosti:")
            lime_image = gr.Image(type="pil", label="LIME Objašnjenje")

    # Shared state for prediction result
    prediction_state = gr.State(value=None)

    # Bind the "Očisti unos" button to the clear_inputs function
    clear_button.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[input_image, pred_image, confidence_text, lime_image, lime_button]
    )

    # Bind buttons to actions
    predict_button.click(
        fn=lambda image: (*detect_and_classify(image), image, gr.update(visible=True)),
        inputs=input_image,
        outputs=[pred_image, confidence_text, prediction_state, lime_button],
    )

    lime_button.click(
        fn=lambda state_image: lime_explanation(state_image) if state_image is not None else None,
        inputs=prediction_state,
        outputs=[lime_image],
    )

# Launch Gradio app
iface.launch(share=True)

# iface = gr.Interface(
#     fn=lambda image: (*detect_and_classify(image), lime_explanation(image)),
#     inputs=gr.Image(type="pil", label="Prenesite sliku"),
#     outputs=[
#         gr.Image(type="pil", label="Prepoznata gljiva"),
#         gr.Textbox(label="Postoci pouzdanosti:"),
#         gr.Image(type="pil", label="LIME Vizualizacija")],
#     title="Sustav za prepoznavanje gljiva",
#     description=(
#         "<br><br>"
#         "<h1 style='color:red;'>!!! UPOZORENJE !!!</h1>"
#         "<p style='font-size: 18px;'>Ova aplikacija koristi model umjetne inteligencije za prepoznavanje gljiva i time može pogriješiti."
#         " Preporučujemo da ne konzumirate gljive na temelju rezultata ove aplikacije bez dodatne provjere.</p>"
#         "<br>"
#         "<p style='font-size: 18px;'>Model može prepoznati 9 različitih vrsta gljiva navedenih u legendi, a uz predikciju navedena je i pouzdanost modela.</p>"
#         "<div style='margin-top: 20px; font-size: 16px;'>"
#         "<h3 style='font-size: 20px;'>Legenda:</h3>"
#         "<div style='display: flex; align-items: center; margin-bottom: 5px;'>"
#         "    <div style='width: 25px; height: 25px; background-color: red; margin-right: 10px;'></div>"
#         "    <span>Otrovne gljive - Koprenka, Krasnica, Muhara, Rudoliska</span>"
#         "</div>"
#         "<div style='display: flex; align-items: center; margin-bottom: 5px;'>"
#         "    <div style='width: 25px; height: 25px; background-color: green; margin-right: 10px;'></div>"
#         "    <span>Jestive gljive - Pečurka, Rujnica, Slinavka, Vrganj</span>"
#         "</div>"
#         "<div style='display: flex; align-items: center;'>"
#         "    <div style='width: 25px; height: 25px; background-color: blue; margin-right: 10px;'></div>"
#         "    <span>Zaštićene gljive - Vlažnica</span>"
#         "</div>"
#         "</div>"
#     ),
# )
# # Launch Gradio app with public URL
# iface.launch(share=True, debug = True)
