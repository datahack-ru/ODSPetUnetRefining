import logging
import numpy as np
import tensorflow as tf
import random
import gc
# import objgraph

import broccole.augmentations as augmentations
from broccole.logUtils import usedMemory

logger = logging.getLogger(__name__)

class HumanDatasetSequence(tf.keras.utils.Sequence):
    def __init__(self, humanDataset, nonHumanDataset, packetSize: int, preprocess_input = None):
        self.humanDataset = humanDataset
        self.humanDataset.reset()
        self.nonHumanDataset = nonHumanDataset
        self.nonHumanDataset.reset()
        self.packetSize = packetSize
        self.nonHumanPacketSize = max((packetSize * len(nonHumanDataset)) // len(humanDataset), 1)
        self.preprocess_input = preprocess_input
        self.elementsReadCount = 0
        self.packetsCount = len(self.humanDataset) // self.packetSize

    def __len__(self):
        return self.packetsCount
    
    def __getitem__(self, index: int):
        imageSize = 224
        
        logger.debug('reading %d batch, memory used %f', index, usedMemory())
        self.humanDataset.index = index * self.packetSize
        x_train_h, y_train_h = self.humanDataset.readBatch(self.packetSize)
        x_train_h, y_train_h = augmentations.appendTransforms(x_train_h, y_train_h, augmentations.train_transforms_after_resize, augmentations.resize_transforms(imageSize))
        logger.debug('reading human batch, memory used %f', usedMemory())
        self.nonHumanDataset.index = index * self.nonHumanPacketSize
        x_train_nh, y_train_nh = self.nonHumanDataset.readBatch(self.nonHumanPacketSize)
        x_train_nh, y_train_nh = augmentations.appendTransforms(x_train_nh, y_train_nh, augmentations.train_transforms_after_resize, augmentations.resize_transforms(imageSize))
        logger.debug('reading nonHuman batch, memory used %f', usedMemory())
        self.elementsReadCount += self.packetSize + self.nonHumanPacketSize
        x_train, y_train = HumanDatasetSequence.shuffleHumanNonHuman(x_train_h, x_train_nh, y_train_h, y_train_nh)
        x_train = np.concatenate((x_train, ))
        y_train = np.concatenate((y_train, ))

        if self.preprocess_input is not None:
            x_train = self.preprocess_input(x_train)
            logger.debug('preprocess x_train, memory used %f', usedMemory())

        del x_train_h
        del x_train_nh
        del y_train_h
        del y_train_nh

        gc.collect()
        # objgraph.show_most_common_types(limit=50)
        # obj = objgraph.by_type('list')[1000]
        # objgraph.show_backrefs(obj, max_depth=10)

        logger.debug('concatenate batches, memory used %f', usedMemory())
        return x_train, y_train

    @staticmethod
    def shuffleHumanNonHuman(humanImages, nonHumanImages, humanMasks, nonHumanMasks):
        totalLength = len(humanImages) + len(nonHumanImages)
        indices = list(range(totalLength))
        random.shuffle(indices)
        imagesPacket = humanImages + nonHumanImages
        masksPacket = humanMasks + nonHumanMasks
        for index in indices:
            permitationIndex = indices[index]
            if permitationIndex < len(humanImages):
                imagesPacket[index] = humanImages[permitationIndex]
                masksPacket[index] = humanMasks[permitationIndex]
            else:
                permitationIndex -= len(humanImages)
                imagesPacket[index] = nonHumanImages[permitationIndex]
                masksPacket[index] = nonHumanMasks[permitationIndex]
        return imagesPacket, masksPacket
