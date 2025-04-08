import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load trained model
MODEL_PATH = "CNN_30ep_32bsize_5000item.keras"
model = load_model(MODEL_PATH)

# Class mapping (ensure this matches your training labels)
class_mapping = {
    0: 'ant', 1: 'bear', 2: 'bee', 3: 'bird', 4: 'butterfly', 5: 'camel',
    6: 'cat', 7: 'crab', 8: 'crocodile', 9: 'dog', 10: 'dolphin', 11: 'fish',
    12: 'frog', 13: 'giraffe', 14: 'horse', 15: 'kangaroo', 16: 'lion',
    17: 'octopus', 18: 'panda', 19: 'penguin', 20: 'sea turtle',
    21: 'shark', 22: 'snail', 23: 'tiger'
}

# List of animals for user to choose from
animal_options = list(class_mapping.values())

IMG_SIZE = 64  # Update this if you trained on a different image size

# Streamlit UI
st.title("🖌️ AI-Powered Animal Sketch Recognizer")
st.write("Draw an animal and let the AI predict what it is!")

# Dropdown menu to choose an animal to draw
chosen_animal = st.selectbox("Choose an animal to draw:", animal_options)

# Create a drawing canvas
canvas = st_canvas(
    fill_color="black",
    stroke_width=2,
    stroke_color="white",
    background_color="black",
    width=600,
    height=600,
    drawing_mode="freedraw",
    key="canvas",
)


# Function to preprocess the image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((64, 64))  # Resize to 64x64
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = image.reshape(1, 64, 64, 1)  # Add batch dimension
    return image

# Process the image when "Guess" is clicked
if st.button("Guess"):
    if canvas.image_data is not None:
        # Convert drawing to an image
        image = Image.fromarray((canvas.image_data[:, :, :3] * 255).astype(np.uint8))
        
        # Preprocess image
        processed_image = preprocess_image(image)

        # Predict using model
        prediction = model.predict(processed_image)
        print(prediction)
        predicted_class = np.argmax(prediction)  # Get the highest probability class
        predicted_animal = class_mapping[predicted_class]  # Map to animal name
        
        # Display the result
        st.write(f"🤖 AI Guess: **{predicted_animal}**")