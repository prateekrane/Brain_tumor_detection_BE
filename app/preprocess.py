import numpy as np
import cv2
from PIL import Image

def preprocess_image(file):
    # Convert uploaded file to grayscale
    img = Image.open(file).convert('L')  # Convert to grayscale
    img = img.resize((128, 128))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 128, 128, 1)  # Add batch and channel dimensions
    return img_array
