
from keras.applications import VGG19
from keras.layers import Input
from keras.models import Model

def define_vgg():
    """
    Builds a pre-trained VGG19 model that outputs image features extracted at the
    third block of the model
    """
    vgg = VGG19(include_top=False, weights="imagenet",input_shape=(128,128,3))
    # Set outputs to outputs of last conv. layer in block 3
    # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
    #vgg.outputs = [vgg.layers[9].output]
    #img = Input(shape=(128,128,3))
    # Extract image features
    #img_features = vgg(img)
    return Model([vgg.input], [vgg.layers[9].output], name='vgg19')