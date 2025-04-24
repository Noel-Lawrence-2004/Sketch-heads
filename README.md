# Sketch-heads

An interactive web app that lets users **draw animal sketches** and uses a **Convolutional Neural Network (CNN)** to guess what the sketch represents! Built using **TensorFlow**, **Streamlit**, and a custom-trained model on a dataset of 64x64 black-and-white animal drawings.

---

## ✨ Demo

Draw a sketch on the canvas and click **"Guess"** to let the AI predict the animal!

> 🐝 Example prediction: "bee" with 92% confidence.

---

## 🚀 Features

- 🎨 Intuitive freehand drawing canvas (powered by `streamlit-drawable-canvas`)
- 🧠 Deep learning prediction engine (trained CNN model)
- 📈 Top-N predictions with confidence scores
- 🔎 Real-time image preprocessing visualization
- ⚙️ Smart image preprocessing: grayscale, contrast boost, resizing, inversion

---

## 🧰 Tech Stack

| Technology | Description |
|------------|-------------|
| [TensorFlow](https://www.tensorflow.org/) | Deep learning framework for training and inference |
| [Streamlit](https://streamlit.io/) | Web app framework for creating UIs with Python |
| [NumPy](https://numpy.org/) | Array processing and image handling |
| [PIL (Pillow)](https://pillow.readthedocs.io/) | Image processing (grayscale, inversion, resize) |

---

## 📁 Project Structure

```bash
├── app.py                     # Main Streamlit app
├── CNN_model_128bs.keras      # Trained TensorFlow model
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation (you’re here!)
