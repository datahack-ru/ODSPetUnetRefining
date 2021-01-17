import os
import argparse
import cv2
import albumentations as albu
import logging
import numpy as np

from broccole.logUtils import init_logging
import broccole.augmentations as augmentations
from broccole.SegmentationDataset import SegmentationDataset

logger = logging.getLogger(__name__)


def augmentationTest(transform):
    imageSize = 224
    camera = cv2.VideoCapture(0)
    while True:
        ret_val, image = camera.read()
        if image is None:
            continue
        mask = np.zeros_like(image)
        cv2.circle(mask, (mask.shape[1] // 2, mask.shape[0] // 2), mask.shape[0] // 4, (255, 255, 255), 5)
        cv2.circle(mask, (mask.shape[1] // 2 + 5, mask.shape[0] // 2 + 10), mask.shape[0] // 8, (255, 255, 255), 1)
        images, masks = augmentations.appendTransforms([image], [mask], transform, augmentations.resize_transforms(imageSize))
        for i in range(len(images)):
            cv2.imshow('image{}'.format(i), images[i])
            cv2.imshow('mask{}'.format(i), masks[i])
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def datasetAugmentationTest(transform, dataset):
    imageSize = 224
    packetSize = 1

    packets = len(dataset) // packetSize
    for packetIndex in range(packets):
        images, masks = dataset.readBatch(packetSize)
        images, masks = augmentations.appendTransforms(images, masks, transform, augmentations.resize_transforms(imageSize))
        for i in range(len(images)):
            cv2.imshow('image{}'.format(i), images[i])
            cv2.moveWindow('image{}'.format(i), images[i].shape[1] * 2, images[i].shape[0] * i * 2)
            cv2.imshow('mask{}'.format(i), masks[i])
            cv2.moveWindow('mask{}'.format(i), 0, masks[i].shape[0] * i * 2)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description='test augmentations')
    parser.add_argument('--datasetDir', help='path to directory with dataset', type=str)    
    args = parser.parse_args()
    return args

def main():
    init_logging('augmentationTest.log')

    args = parse_args()
    datasetDir = args.datasetDir

    black = (0, 0, 0)
    # transform = albu.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=black, mask_value=black, p=1)

    # transform = albu.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=black, mask_value=black, p=1)
    # transform = albu.MedianBlur(p=1)
    # transform = albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, value=black, mask_value=black, p=1)
    transform = augmentations.train_transforms_after_resize
    # augmentationTest(transform)

    humanDataset = SegmentationDataset(os.path.join(datasetDir, 'human'), 61600, shuffle=True)
    # valHumanDataset = SegmentationDataset(os.path.join(datasetDir, 'valHuman'), 2693, shuffle=True)
    datasetAugmentationTest(transform, humanDataset)

if __name__ == '__main__':
    main()
