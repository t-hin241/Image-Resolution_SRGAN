from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

#######Params##########
CHANNELS = 3
LR_HEIGHT = 32
LR_WIDTH = 32
LR_SHAPE = (LR_HEIGHT, LR_WIDTH, CHANNELS)
HR_HEIGHT = LR_HEIGHT*4
HR_WIDTH = LR_WIDTH*4
HR_SHAPE = (HR_HEIGHT, HR_WIDTH, CHANNELS)

n_resblocks = 16
gen_filters = 64
disc_filters = 64

#######Generator#############


def residual_block(layer_input, filters):
    """Residual block described in paper"""
    d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
    d = Activation('relu')(d)
    d = BatchNormalization(momentum=0.8)(d)
    d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
    d = BatchNormalization(momentum=0.8)(d)
    d = Add()([d, layer_input])
    return d

def deconv2d(layer_input):
    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
    u = Activation('relu')(u)
    return u

def define_generator():
    # Low resolution image input
    img_lr = Input(shape=LR_SHAPE)

    # Pre-residual block
    c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
    c1 = Activation('relu')(c1)

    # Propogate through residual blocks
    r = residual_block(c1, gen_filters)
    for _ in range(n_resblocks - 1):
        r = residual_block(r, gen_filters)

    # Post-residual block
    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
    c2 = BatchNormalization(momentum=0.8)(c2)
    c2 = Add()([c2, c1])

    # Upsampling
    u1 = deconv2d(c2)
    u2 = deconv2d(u1)

    # Generate high resolution output
    gen_hr = Conv2D(CHANNELS, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

    return Model(img_lr, gen_hr)



##########Discriminator############

def d_block(layer_input, filters, strides=1, bn=True):
    """Discriminator layer"""
    d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
        d = BatchNormalization(momentum=0.8)(d)
    return d

def define_discriminator():
    # Input img
    d0 = Input(shape=HR_SHAPE)

    d1 = d_block(d0, disc_filters, bn=False)
    d2 = d_block(d1, disc_filters, strides=2)
    d3 = d_block(d2, disc_filters*2)
    d4 = d_block(d3, disc_filters*2, strides=2)
    d5 = d_block(d4, disc_filters*4)
    d6 = d_block(d5, disc_filters*4, strides=2)
    d7 = d_block(d6, disc_filters*8)
    d8 = d_block(d7, disc_filters*8, strides=2)

    d9 = Dense(disc_filters*16)(d8)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)
    return Model(d0, validity)


##########Combination#########
def build_combined(vgg, gen, disc):
    # High res. and low res. images                                          #combined model input
    img_hr = Input(shape=HR_SHAPE) 
    img_lr = Input(shape=LR_SHAPE)
    # Generate high res. version from low res.                               #path to output
    fake_hr = gen(img_lr) 

    # Extract image features of the generated img                            #path to output
    fake_features = vgg(fake_hr)

    # For the combined model we will only train the generator            
    disc.trainable = False

    # Discriminator determines validity of generated high res. images       #path to output
    validity = disc(fake_hr)
    combined = Model([img_lr, img_hr], [validity, fake_features])        # Model([input1,input2],[output1,output2])        
    combined.compile(loss=['binary_crossentropy', 'mse'],                # output1 vs groundtruth1  (label) : binaray_crossentropy loss
                          loss_weights=[1e-3, 1],                        # output2 vs groundtruth2  (feature) : mse loss
                          optimizer=Adam(0.0002, 0.5))
    
    return combined