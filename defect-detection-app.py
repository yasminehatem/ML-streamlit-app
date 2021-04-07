import streamlit as st
import numpy as np
from PIL import Image
"""
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow import keras
from tensorflow.keras.applications import vgg16
"""


#model = keras.models.load_model('model.h5')
#model = vgg16.VGG16(weights='imagenet')


st.title('Metal Surface Defect Detection')

file_up = st.file_uploader("Upload an image", type="jpg")


if file_up is not None:
  image = Image.open(file_up)
  st.image(image, caption='Uploaded Image.', use_column_width=True)
  """PIL_image = load_img(image, target_size=(224, 224))
  numpy_image = img_to_array(PIL_image) #(224, 224, 3)
  # Thus we add the extra dimension to the axis 0.
  image_batch = np.expand_dims(numpy_image, axis=0) #(1, 224, 224, 3)
  processed_image = model.preprocess_input(image_batch.copy())
  pred_button = st.button("Predict")

  if pred_button:
    prediction = model.predict(processed_image)
    st.write('result: %s' % prediction)"""
    
