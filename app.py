import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

#Load the trained model
model = load_model('image_classification_model.h5')

#Define your class labels (update to your dataset)
CLASS_NAMES = ['bike', 'bus', 'car', 'cat', 'dog']

#Function to preprocess the uploaded image
def preprocess_image(img):
    #Convert RGBA or grayscale to RGB
    #rgba has 4 channel whereas rgb has 3 channel thats why getting error
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((224, 224))  # Resize image to match model input
    img_array = image.img_to_array(img) / 255.0 
    return np.expand_dims(img_array, axis=0)  

#Streamlit UI
st.set_page_config(page_title="Real-Time Image Classifier")
st.title("Image Classification App")
st.write("Upload an image and see what the model predicts!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    #process iage
    img_array = preprocess_image(img)

    #Make predictions
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    #Display the prediction
    st.markdown("---")
    st.subheader(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    #Show probability bar chart
    st.bar_chart(predictions[0])
