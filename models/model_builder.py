from tensorflow import keras
from models.model import csnet_model

def seg_model_build(image_size):

    input, output = csnet_model(weights=None, input_tensor=None, input_shape=(image_size[0], image_size[1], 3),
                                    classes=2, OS=16)
    model = keras.Model(input, output)
    model.summary()
    return model