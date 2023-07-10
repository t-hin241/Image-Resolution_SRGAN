from srgan import define_generator
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
from skimage.transform import resize
import numpy as np
from utils import sample_image
model = define_generator()
model.load_weights("./generator_weights/gweight999.hdf5")
org_img = imread("./test_imgs/image1.jpg")

#org_img = org_img.real.astype(float)
im = resize(org_img,(32,32))
input = np.array([im])
pred = model.predict(input)[-1]
sample_image(im, pred)