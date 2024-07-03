import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image


# Load the pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Function to preprocess an image for the ResNet50 model

def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize the image to (224, 224)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Function to classify an image using the ResNet50 model
def classify_image(img_path):
    preprocessed_img = preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Function to check if the image belongs to the accepted classes
def get_image_category(predictions):
    for _, label, _ in predictions:
        if label.lower() == 'parkinson':
            return 'Parkinson'
        elif label.lower() == 'healthy':
            return 'Healthy'
    return None

# Function to handle the uploaded image
def handle_uploaded_image(uploaded_image_path, trained_parkinson_path, trained_healthy_path, test_parkinson_path, test_healthy_path):
    try:
        # Load trained and test image filenames
        trained_parkinson_images = os.listdir(trained_parkinson_path)
        trained_healthy_images = os.listdir(trained_healthy_path)
        test_parkinson_images = os.listdir(test_parkinson_path)
        test_healthy_images = os.listdir(test_healthy_path)

        # Get the predictions for the uploaded image
        predictions = classify_image(uploaded_image_path)
        category = get_image_category(predictions)

        # Check if the image belongs to trained or test categories
        if os.path.basename(uploaded_image_path) in trained_parkinson_images:
            print("Image belongs to the trained Parkinson set.")
        elif os.path.basename(uploaded_image_path) in trained_healthy_images:
            print("Image belongs to the trained Healthy set.")
        elif os.path.basename(uploaded_image_path) in test_parkinson_images:
            print("Image belongs to the test Parkinson set.")
        elif os.path.basename(uploaded_image_path) in test_healthy_images:
            print("Image belongs to the test Healthy set.")
        elif category:
            print(f"The image belongs to: {category}")
        else:
            print("Error: Image doesn't match accepted categories.")
    except Exception as e:
        print(f"Error processing the image: {e}")

# Input paths
trained_parkinson_path = r'D:\all\project\code\mod code\project code\CODE\Backend\Data\train\parkinson'
trained_healthy_path = r'D:\all\project\code\mod code\project code\CODE\Backend\Data\train\healthy'
test_parkinson_path = r'D:\all\project\code\mod code\project code\CODE\Backend\Data\test\parkinson'
test_healthy_path =r'D:\all\project\code\mod code\project code\CODE\Backend\Data\test\healthy'
uploaded_image_path = r'D:\all\project\code\mod code\project code\CODE\re.png'

# Example usage
handle_uploaded_image(uploaded_image_path, trained_parkinson_path, trained_healthy_path, test_parkinson_path, test_healthy_path)
