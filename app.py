import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(
    page_title="COVID-19 Chest X-Ray Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define a function to load the model
@st.cache_resource
def load_prediction_model():
    try:
        model = tf.keras.models.load_model('Model_for_Covid19_Prediction.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_prediction_model()

# Title and description
st.title("COVID-19 Chest X-Ray Predictor")
st.write("""
### Upload a chest X-ray image to predict if the patient is COVID-19 positive or negative.
This model is based on a Convolutional Neural Network (CNN) trained using TensorFlow.
""")

# Sidebar information
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload a chest X-ray image (JPEG or PNG format).
2. Click on the **Predict** button to see the result.
3. Use the **Clear** button to reset the app.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Button to predict
if st.button("Predict"):
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Chest X-Ray', use_column_width='auto', width=300)
        st.write("Classifying...")

        # Load image and preprocess for prediction
        def preprocess_image(image_file):
            img = image.load_img(image_file, target_size=(150, 150))  # Resize image
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array /= 255.0  # Normalize to [0,1] range
            return img_array

        # Preprocess the uploaded image
        img_array = preprocess_image(uploaded_file)

        # Make prediction if model is loaded
        if model is not None:
            prediction = model.predict(img_array)
            prediction_label = 'COVID-19 Positive' if prediction < 0.5 else 'COVID-19 Negative'
            confidence = (1 - prediction[0][0]) if prediction_label == 'COVID-19 Positive' else prediction[0][0]

            # Display prediction result
            st.write(f"### Prediction: **{prediction_label}**")
            st.write(f"### Confidence: **{confidence:.2f}**")

            # Plot the probability of both classes with smaller size
            fig, ax = plt.subplots(figsize=(4, 2))
            labels = ['COVID-19 Positive', 'COVID-19 Negative']
            probabilities = [1 - prediction[0][0], prediction[0][0]]
            ax.barh(labels, probabilities, color=['#FF6347', '#4682B4'])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Prediction Probability', fontsize=10)
            st.pyplot(fig)

            # Example confusion matrix (dummy values for demonstration)
            true_labels = [0, 1]  # Assuming 0 is Negative and 1 is Positive
            pred_labels = [1 if p < 0.5 else 0 for p in probabilities]

            # Confusion matrix
            confusion_mtx = tf.math.confusion_matrix(true_labels, pred_labels).numpy()
            fig3, ax3 = plt.subplots(figsize=(4, 4))
            sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', ax=ax3, 
                        xticklabels=['Negative', 'Positive'], 
                        yticklabels=['Negative', 'Positive'])
            ax3.set_xlabel('Predicted Labels')
            ax3.set_ylabel('True Labels')
            ax3.set_title('Confusion Matrix', fontsize=10)
            st.pyplot(fig3)

    else:
        st.warning("Please upload a chest X-ray image to get a prediction.")

# Clear button functionality
if st.button("Clear"):
    st.session_state.clear()
    st.success("Inputs cleared. Please upload a new image.")

# Footer
st.markdown("""
---
*This application is developed using Streamlit and a TensorFlow-based deep learning model.*
""")
