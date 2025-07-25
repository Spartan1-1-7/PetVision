import numpy as np
from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model

def predict_img(img_path):
    model=load_model('dog_cat_model.h5')
    test_image=load_img(img_path,target_size=(64, 64))
    test_image=img_to_array(test_image)
    test_image=np.expand_dims(test_image, axis=0)
    result= model.predict(test_image)
    if result[0][0] == 1:
        prediction='dog'
    else:
        prediction='cat'
    return prediction