import streamlit as st
import numpy as np
from PIL import Image

from tensorflow import keras
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions



#model = keras.models.load_model('model.h5')
model = vgg16.VGG16(weights='imagenet')

st.title('Metal Surface Defect Detection')
file_up = st.file_uploader("Upload an image", type="jpg")

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    #st.write("")
    #st.write("Classifying...")
    PIL_image = image.resize((224, 224))
    #PIL_image = load_img(image, target_size=(224, 224))
    numpy_image = img_to_array(PIL_image) #(224, 224, 3)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0) #(1, 224, 224, 3)
    processed_image = vgg16.preprocess_input(image_batch.copy())
    pred_button = st.button("Predict")

    if pred_button:
        prediction = model.predict(processed_image)
        label = decode_predictions(prediction)
        # return highest probability 
        label = label[0][0]
        st.write('%s (%.2f%%)' % (label[1], label[2]*100))
    

