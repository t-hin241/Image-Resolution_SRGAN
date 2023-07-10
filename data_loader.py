from matplotlib.pyplot import imread
from skimage.transform import resize
from glob import glob
import numpy as np
import os

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        folder_path = "./%s" % self.dataset_name
        paths = glob('./%s/*' % (self.dataset_name))

        batch_images = np.random.choice(paths, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:

            img = self.read(img_path)
            #print(img.shape)

            h, w = self.img_res
            low_h, low_w = int(h / 4), int(w / 4)

            img_hr = resize(img, self.img_res)
            img_lr = resize(img, (low_h, low_w))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr)
        imgs_lr = np.array(imgs_lr)

        return imgs_hr, imgs_lr


    def read(self, path):
        return imread(path).real.astype(float)