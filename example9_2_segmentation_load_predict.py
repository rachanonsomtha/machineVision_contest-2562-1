from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os


IMAGE_SIZE = (512, 512)


def iou(y_true, y_pred):
    y_true = K.cast(K.greater(y_true, 0.5), dtype='float32')
    y_pred = K.cast(K.greater(y_pred, 0.5), dtype='float32')
    inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = K.sum(
        K.sum(K.squeeze(K.clip(y_true + y_pred, 0, 1), axis=3), axis=2), axis=1)
    return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

# thresholding output image func
# def threshold_slow(T, image):
#     h = image.shape[0]
#     w = image.shape[1]
#     for y in range(0, h):
#         for x in range(0, w):
#             image[y, x] = 255 if image[y, x] >= T else 0
#     return image


#load trained model
model = load_model('my_model_final.h5', custom_objects={'iou': iou})

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', iou])

# print(model.summary())
# test_im = cv2.imread('textlocalize/validation/Input/045.jpg')


#folder contain test images
testfileloc = list(glob.glob('textlocalize/validation/contest/Input/*'))
# print(testfileloc)


# true_size = test_im.shape
# imshow_size = (512, round(true_size[0]*512/true_size[1]))
# cv2.imshow('Input', cv2.resize(k, imshow_size))

## destinate output folder path
path = 'textlocalize/validation/test_output_final1'

L = 2
kernel = np.ones((L, L), np.float32) / L / L
for k in range(len(testfileloc)):
    filename = str(os.path.basename(testfileloc[k])) ## get filename
    # print(filename)
    test_im = cv2.imread(testfileloc[k])
    true_size = test_im.shape

    test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
    test_im = cv2.resize(test_im, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    test_im = test_im/255.
    test_im = np.expand_dims(test_im, axis=0)

    segmented = model.predict(test_im)
    #segmented = np.around(segmented)
    segmented = (segmented[0, :, :, 0]*255).astype('uint8')

    # segmented = cv2.erode(segmented, kernel)

    segmented[segmented >128] = 255
    segmented[segmented <= 128] = 0


    # segmented  = cv2.erode(segmented, kernel )
    # cv2.imshow('show', cv2.resize(segmented, (true_size[1], true_size[0])))
    # cv2.waitKey()
    if(cv2.imwrite(os.path.join(path, filename), cv2.resize(segmented, (true_size[1], true_size[0])))):
        print('No: ' + str(k +1)+ ' ' + filename + ' is saved')
    # cv2.waitKey()
    # cv2.imwrite('%s.jpg'%filename, segmented)
