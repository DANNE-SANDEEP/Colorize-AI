import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image

# Set paths to model files
DIR = r"C:/Personal/LPU/Machine Learning - II/Colorization"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# Load the model
@st.cache_resource(show_spinner=False)
def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    # Load centers for ab channel quantization
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    return net

# Function to colorize image
def colorize_image(image, net):
    image = np.array(image.convert("RGB"))
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return colorized

# Streamlit app UI
def main():
    st.title("Black and White Image Colorizer")

    # Project Description
    st.markdown("""
    ### Project Overview
    This application uses a **pre-trained deep learning model** to automatically colorize black and white images. 
    The model is based on the research from Richard Zhang et al., and it employs a neural network trained on millions 
    of images to predict colors for black and white photos. The key steps involved in this process are:
    
    - Upload a black and white image.
    - The app uses a deep learning model to add color to the image.
    - The original and colorized images are displayed side by side.

    You can try uploading your own images to see how well the model colorizes them!
    """)

    # Upload the image
    uploaded_image = st.file_uploader("Upload a black and white image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display input and output side by side using columns
        col1, col2 = st.columns(2)

        # Display the original image in the first column
        with col1:
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Black and White Image", use_column_width=True)

        # Load the model
        net = load_model()

        # Show a spinner while the image is being colorized
        with st.spinner("Adding colors to the image..."):
            # Colorize the image
            colorized_image = colorize_image(image, net)

        # Display the colorized image in the second column
        with col2:
            st.image(colorized_image, caption="Colorized Image", use_column_width=True)

if __name__ == "__main__":
    main()
