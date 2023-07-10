
import numpy as np
from data_loader import DataLoader
from srgan import define_generator, define_discriminator, build_combined
from vgg import define_vgg
from keras.optimizers import Adam
from utils import sample_image

####Params######
epochs = 200
batch_size = 50
disc_patch = (8,8,1) #hr_height / 2**4
sample_interval = 50
gen_path = "./generator_weights/gweight%d.hdf5"
disc_path = "./discriminator_weights/dweight%d.hdf5"

####dataset####
data_loader = DataLoader("data folder name")

####Model######
vgg = define_vgg()
vgg.trainable = False
vgg.compile(loss='mse',
            optimizer=Adam(0.0002, 0.5),
            metrics=['accuracy'])

generator = define_generator()

discriminator = define_discriminator()
discriminator.compile(loss='mse',
            optimizer=Adam(0.0002, 0.5),
            metrics=['accuracy'])

combined = build_combined(vgg, generator, discriminator)



for epoch in range(epochs):
    # ----------------------
    #  Train Discriminator
    # ---------------------
    # Sample images and their conditioning counterparts
    imgs_hr, imgs_lr = data_loader.load_data(batch_size)
    # From low res. image generate high res. version
    fake_hr = generator.predict(imgs_lr)
    valid = np.ones((batch_size,) + disc_patch)
    fake = np.zeros((batch_size,) + disc_patch)
    # Train the discriminators (original images = real / generated = Fake)
    d_loss_real = discriminator.train_on_batch(imgs_hr, valid)
    d_loss_fake = discriminator.train_on_batch(fake_hr, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # ------------------
    #  Train Generator
    # -----------------
    # Sample images and their conditioning counterparts
    imgs_hr, imgs_lr = data_loader.load_data(batch_size)
    # The generators want the discriminators to label the generated images as real
    valid = np.ones((batch_size,) + disc_patch)
    # Extract ground truth image features using pre-trained VGG19 model
    image_features = vgg.predict(imgs_hr)
    # Train the generators
    g_loss = combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])               # [Y1,Y2] = [groundtruth1,groundtruth2] = [valid, image_features]

    # Plot the progress
    print ("Epoch %d: Gen Loss: %s -- Disc Loss: %s" % (epoch, g_loss[0], d_loss[0]))
    # If at save interval => save generated image samples
    if epoch % sample_interval == 0 or epoch==999:
        generator.save(gen_path % epoch)
        discriminator.save(disc_path % epoch)
