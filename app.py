import gradio as gr
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('safae_tuberculosis_detection_lenet_model.keras')  # Replace with your model path

# Preprocessing function
def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')  # Convert NumPy array to PIL image
    image = image.convert("RGB")  # Convert grayscale to RGB
    image = image.resize((256, 256))  # Resize image to 256x256 pixels
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function
def predict(image):
    image = preprocess_image(image)
    prediction = model.predict(image)
    return "Tuberculosis" if prediction > 0.5 else "Normal"

# Gradio interface function
def classify_image(image):
    result = predict(image)
    return result

# Create Gradio interface
title = "Tuberculosis Detection"
description = "Upload a chest X-ray image to detect if it indicates Tuberculosis."
examples = [["example_image.png"]]  # Provide a path to an example image if available

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(image_mode='RGB'),
    outputs=gr.Textbox(label="Prediction"),
    title=title,
    description=description,
    examples=examples
)

if __name__ == "__main__":
    interface.launch()
