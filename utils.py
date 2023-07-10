from matplotlib import pyplot as plt

def sample_image(img_lr, img_hr, epoch=1, fname='result'):
    r, c = 1, 2

    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(r, c, 1)
    # showing image
    plt.imshow(img_lr)
    plt.axis('off')
    plt.title("LR")
    # Adds a subplot at the 2nd position
    fig.add_subplot(r, c, 2)
    # showing image
    plt.imshow(img_hr)
    plt.axis('off')
    plt.title("HR")

    fig.savefig("./images/%s%d.png" % (fname,epoch))
    plt.close()