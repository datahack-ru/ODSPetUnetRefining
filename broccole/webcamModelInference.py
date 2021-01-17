import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import logging
import time

from broccole.Timer import Timer
from broccole.logUtils import init_logging
import broccole.augmentations as augmentations

logger = logging.getLogger(__name__)

def inference(model):
    imageSize = 224
    totalTime = Timer('total time', 10)
    readFrameTime = Timer('read frame', 10)
    inferenceTime = Timer('inference', 10)

    camera = cv2.VideoCapture(0)
    while True:
        totalTime.start()
        readFrameTime.start()
        ret_val, img = camera.read()
        readFrameTime.end()

        inferenceTime.start()
        data = {
            'image': img
        }
        augData = augmentations.valid_transforms(imageSize)(**data)
        imgs = [augData['image']]

        imgs = np.stack(imgs)

        masks = model.predict(imgs)
        inferenceTime.end()

        for i in range(masks.shape[0]):
            image = imgs[i]
            mask = masks[i]
            threshold = np.copy(mask)
            maskThreshold = 0.3 # 150 / 255
            binaryMask = mask[:, :, :] < maskThreshold
            threshold[binaryMask] = 0.0
            threshold[~binaryMask] = 1.0
            ksize = (3, 3)
            threshold = cv2.blur(threshold, ksize)
            imageMask = np.stack((threshold, threshold, threshold), axis=-1)
            background = np.ones_like(image) / 2
            cv2.imshow('image', image * imageMask + background * (1.0 - imageMask))
            cv2.imshow('mask', mask)
            cv2.imshow('threshold', threshold)
            cv2.waitKey(1)
        totalTime.end()

        totalTime.logAverageTime()
        readFrameTime.logAverageTime()
        inferenceTime.logAverageTime()

    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description='inference U-Net on webcam')
    parser.add_argument('--modelFilePath', help='path to load/save model', type=str)
    args = parser.parse_args()
    return args

def main():
    init_logging('webcamInference.log')

    args = parse_args()
    modelFilePath = args.modelFilePath    

    model = tf.keras.models.load_model(
        modelFilePath,
        compile=False
    )

    # model = tf.saved_model.load(modelFilePath)

    print("model summary\n{}".format(model.summary()))

    inference(model)


if __name__ == '__main__':
    main()