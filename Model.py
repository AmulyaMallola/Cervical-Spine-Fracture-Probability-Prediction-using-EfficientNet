import numpy as np
import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
import efficientnet.tfkeras as efficientnet
import tensorflow_hub as hub
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense
from keras.layers import SeparableConv2D
def get_model():
    inp = keras.layers.Input((None, None, 3))
    x = SeparableConv2D(3, 3, padding='SAME')(inp)

    x = efficientnet.EfficientNetB5(include_top=False, weights='imagenet')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    

   
    out = keras.layers.Dense(7, 'softmax')(x)

    
    model = keras.models.Model(inp, out)
    model.summary()

    
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

    return model