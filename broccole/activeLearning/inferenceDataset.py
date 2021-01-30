import os
import cv2
import argparse
import tensorflow as tf
import numpy as np
import shutil

import broccole.augmentations as augmentations

def parse_args():
    parser = argparse.ArgumentParser(description='make video')
    parser.add_argument('--imagesDir', help='path to directory with images', type=str)
    parser.add_argument('--datasetDir', help='path to dataset', type=str)
    parser.add_argument('--modelFilePath', help='path to model', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    imagesDir = args.imagesDir
    datasetDir = args.datasetDir
    modelFilePath = args.modelFilePath

    model = tf.keras.models.load_model(
        modelFilePath,
        compile=False
    )

    imageSize = 224

    imageFilenames = [filename for filename in os.listdir(imagesDir) if os.path.isfile(os.path.join(imagesDir, filename))]
    i = 0
    for imageFilename in imageFilenames:
        ext = os.path.splitext(imageFilename)[1]
        if ext != '.jpg':
            continue
        imagePath = os.path.join(imagesDir, imageFilename)
        print("imagePath {}".format(imagePath))
        sourceImage = cv2.imread(imagePath)

        resized = cv2.resize(sourceImage, (imageSize, imageSize))

        data = {
            'image': sourceImage
        }
        augData = augmentations.valid_transforms(imageSize)(**data)
        imgs = [augData['image']]

        imgs = np.stack(imgs)
        image = imgs[0]

        masks = model.predict(imgs)
        mask = masks[0]

        threshold = np.copy(mask)
        maskThreshold = 0.3 # 150 / 255
        binaryMask = mask[:, :, :] < maskThreshold
        threshold[binaryMask] = 0.0
        threshold[~binaryMask] = 1.0
        ksize = (3, 3)
        blurredThreshold = cv2.blur(threshold, ksize)
        imageMask = np.stack((blurredThreshold, blurredThreshold, blurredThreshold), axis=-1)
        background = np.ones_like(resized) / 2

        shutil.copyfile(imagePath, os.path.join(datasetDir, "image{}.jpg".format(i)))
        cv2.imwrite(os.path.join(datasetDir, "infMask{}.png".format(i)), threshold)

        cv2.imshow('image', resized)
        cv2.imshow('masked image', resized / 255 * imageMask + background * (1.0 - imageMask))
        cv2.imshow('mask', mask)
        cv2.waitKey(1)
        i += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()