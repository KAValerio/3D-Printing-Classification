import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from sklearn import model_selection
from skimage.feature import hog
from sklearn import metrics
import os
from random import randint
from sklearn.linear_model import SGDClassifier
import pickle


##### Adjustable Parameters #####
predvid = 6  # Which video to make a prediction on (numbers 1 through 6)
ppc = 12  # HOG parameter (pixels per cell)
cpb = 3  # HOG parameter (cells per block)
n = 20 ** 2  # Downsampled size of image
rint = 14 #randint(0, 100)  # Random State
showImages = True #Set to true if you want the images to be shown

def read_img(path, n=10 ** 2):
    """
    Read image and store it as an array, given the image path.
    Returns the 3 dimensional image array.
    """
    img = Image.open(path).convert('L')
    img = img.resize((n, n), Image.ANTIALIAS)
    img_arr = np.array(img, dtype='int32')
    img.close()
    return img_arr


def numericalSort(value):
    import re
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


###### Load Data #####
print("Loading data and preprocessing: ")
dirf = 'data/prints/fail/'
dirp = 'data/prints/okay/'
numtot = 0
for root, dirs, files in os.walk('data/'):
    numtot += len(files)
numtotr = range(round(numtot * 0.1), numtot + round(numtot * 0.1) + 1, round(numtot * 0.1))

images = []
labels = []
hogs = []
hogimage = []
i = 0

# fails
for file in listdir(dirf):
    im = read_img(dirf + file, n=n)
    h, him = hog(im, visualize=True,
                 pixels_per_cell=(ppc, ppc),
                 cells_per_block=(cpb, cpb),
                 block_norm="L2-Hys")
    hogs.append(h)
    hogimage.append(him)
    images.append(im)
    labels.append(1)
    if i in numtotr:
        print(str(round(i / numtot * 100, -1)) + '%')
    i += 1

# passes
for file in listdir(dirp):
    im = read_img(dirp + file, n=n)
    h, him = hog(im, visualize=True,
                 pixels_per_cell=(ppc, ppc),
                 cells_per_block=(cpb, cpb),
                 block_norm="L2-Hys")
    hogs.append(h)
    hogimage.append(him)
    images.append(im)
    labels.append(0)
    if i in numtotr:
        print(str(round(i / numtot * 100, -1)) + '%')
    i += 1

# videos
trainvids = [1, 2, 3, 4, 5, 6]
trainvids.remove(predvid)
d = {1: 22, 2: 30, 3: 25, 4: 37, 5: 32, 6: 15}
for v in trainvids:
    dire = 'data/image_stream/video' + str(v) + '/'
    for j, file in enumerate(sorted(listdir(dire), key=numericalSort)):
        ot = 0
        if j >= d[v]:
            ot = 1
        im = read_img(dire + file, n=n)
        h, him = hog(im, visualize=True,
                     pixels_per_cell=(ppc, ppc),
                     cells_per_block=(cpb, cpb),
                     block_norm="L2-Hys")
        hogs.append(h)
        hogimage.append(him)
        images.append(im)
        labels.append(ot)
        if i in numtotr:
            print(str(round(i / numtot * 100, -1)) + '%')
        i += 1

corinds = [k for k in range(len(labels)) if labels[k] == 0]
errorinds = [k for k in range(len(labels)) if labels[k] == 1]

# Creating Images
if showImages:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Histogram of Oriented Gradients (hog)")
    ax[0].imshow(hogimage[corinds[0]])
    ax[0].set_title("Passed Example")
    ax[1].imshow(hogimage[errorinds[0]])
    ax[1].set_title("Failed Example")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.show()

if showImages:
    plt.text(.5, .975, "txt", ha='center')
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.axis('off')
        im = images[corinds[i]]
        plt.imshow(im)
    plt.show()
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.axis('off')
        im = hogimage[corinds[i]]
        plt.imshow(im)
    plt.show()
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.axis('off')
        im = images[errorinds[i]]
        plt.imshow(im)
    plt.show()
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.axis('off')
        im = hogimage[errorinds[i]]
        plt.imshow(im)
    plt.axis('off')
    plt.show()
print("Preprocessing complete")

##### Run Model #####
print("Running Model: ")
X = np.array(hogs)
y = labels
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=rint)

sgd_clf = SGDClassifier(random_state=rint, max_iter=1000, tol=1e-4, penalty='l2', loss="log")
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
print('Model Metrics:')
print(metrics.classification_report(y_test, y_pred))

pickle.dump(sgd_clf, open("SGDModel.pkl", 'wb'))
