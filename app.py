import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
from PIL import ImageEnhance, ImageOps
import cv2

# Load trained model
MODEL_PATH = "CNN_model_128bs.keras"
model = keras.models.load_model(MODEL_PATH, compile=False)

# Class mapping (ensure this matches your training labels)
class_mapping = {
    0: 'bee', 1: 'bird', 2: 'butterfly', 3: 'cat', 4: 'crab', 5: 'dog',
    6: 'fish', 7: 'frog', 8: 'horse', 9: 'kangaroo', 10: 'lion', 11: 'octopus',
    12: 'panda', 13: 'penguin', 14: 'sea turtle', 15: 'snail'
}

# List of animals for user to choose from
animal_options = list(class_mapping.values())

IMG_SIZE = 64 

# Streamlit UI
st.title("CNN Based Sketch Recognizer")
st.write("Draw any one animal and let the AI predict what it is!")

st.markdown("### Animals you can draw:")

cols_per_row = 6
rows = (len(animal_options) + cols_per_row - 1) // cols_per_row

for i in range(rows):
    cols = st.columns(cols_per_row)
    for j in range(cols_per_row):
        index = i * cols_per_row + j
        if index < len(animal_options):
            cols[j].write(animal_options[index])


stroke = st.sidebar.slider("🖊️ Stroke Width", 4, 15, 6)
canvas = st_canvas(
    fill_color="white",
    stroke_width=stroke,
    stroke_color="black",
    background_color="white",
    width=600,
    height=600,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_image(image_array):
    # Remove alpha if present
    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # Convert to grayscale
    image = Image.fromarray((image_array * 255).astype(np.uint8)).convert("L")

    # # Invert: white-on-black
    # image = ImageOps.invert(image)

    # Crop to content
    image_np = np.array(image)
    coords = cv2.findNonZero((image_np > 20).astype(np.uint8))  # find non-black pixels
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        image = image.crop((x, y, x + w, y + h))

    # Resize to 64x64, preserving aspect ratio and padding
    image = ImageOps.pad(image, (64, 64), color=0, centering=(0.5, 0.5))

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(4.0)

    # Normalize and reshape
    image = np.array(image, dtype=np.float32) / 255.0
    return image.reshape(1, 64, 64, 1)


if st.button("Guess"):
    if canvas.image_data is not None:        
        # Convert drawing to an image
        image = Image.fromarray((canvas.image_data[:, :, :3] * 255).astype(np.uint8))

        # Preprocess image
        processed_image = preprocess_image(np.array(image))  # Send raw pixel values

        # Predict using model
        prediction = model.predict(processed_image)
        
        # Predict using model
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)  
        predicted_animal = class_mapping[predicted_class]  # Map to animal name
        
        # Display the result
        st.write(f"🤖 AI Guess: **{predicted_animal}**")
    else:
        st.error("Please draw something on the canvas.")
        