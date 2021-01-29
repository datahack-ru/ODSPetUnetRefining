# import segmentation_models as sm
# import objgraph
# import cv2
# import json

import argparse
import os
import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
import gc
import logging
from datetime import datetime
from typing import List
import traceback

from tensorflow.python.keras.callbacks import CSVLogger

from broccole.CocoDatasetBuilder import CocoDatasetBuilder
from broccole.SegmentationDataset import SegmentationDataset
from broccole.HumanDatasetSequence import HumanDatasetSequence
from broccole.model import makeModel
from broccole.logUtils import init_logging, usedMemory
import broccole.augmentations as augmentations

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='train U-Net')
    parser.add_argument('--datasetDir', help='path to directory with dataset', type=str)
    parser.add_argument('--trainingDir', help='path to directory to save models', type=str)
    parser.add_argument('--modelEncoder', required=False, help='model encoder architecture type', type=str,
                        default="resnet18")
    parser.add_argument('--batchSize', help='batch size', type=int, default=8)
    parser.add_argument('--epochs', help='epochs count', type=int, default=1)
    parser.add_argument('--startEpoch', help='start epoch', type=int, default=0)
    parser.add_argument('--learningRate', help='learning rate', type=float, default=3e-4)#0.001)
    parser.add_argument('--checkpointFilePath', help='path to checkpoint', type=str)
    args = parser.parse_args()
    return args


def save_model(model, trainingDir, modelEncoder, packetIndex):
    now = datetime.now()
    weightsPath = os.path.join(trainingDir,
                               "u-net-{}_{}_{}-{}-{}_{}.chpt".format(modelEncoder, now.day, now.hour, now.minute,
                                                                     now.second, packetIndex))
    model.save_weights(weightsPath)
    logger.info('model saved at %s', weightsPath)


def train(
        model, preprocess_input,
        humanDataset: SegmentationDataset,
        nonHumanDataset: SegmentationDataset,
        valHumanDataset: SegmentationDataset,
        valNonHumanDataset: SegmentationDataset,
        trainingDir: str,
        modelEncoder: str,
        batchSize: int = 1,
        epochs: int = 1,
        startEpoch: int = 0
):
    imageSize = 224

    validationPacketSize = 16 * 16
    x_val_h, y_val_h = valHumanDataset.readBatch(validationPacketSize)
    augmentations.applyTransforms(x_val_h, y_val_h, augmentations.valid_transforms(imageSize))
    x_val_nh, y_val_nh = valNonHumanDataset.readBatch(validationPacketSize)
    augmentations.applyTransforms(x_val_nh, y_val_nh, augmentations.valid_transforms(imageSize))
    x_val = np.stack(x_val_h + x_val_nh)
    y_val = np.stack(y_val_h + y_val_nh)
    x_val = preprocess_input(x_val)

    checkPointPath = os.path.join(trainingDir, 'u-net-{}.chpt'.format(modelEncoder))
    checkPointCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkPointPath,
                                                            verbose=1)

    packetSize = 8 * 8

    humanDatasetSequence = HumanDatasetSequence(humanDataset, nonHumanDataset, packetSize, preprocess_input)

    model.fit(
        x=humanDatasetSequence,
        batch_size=batchSize,
        shuffle=False,
        epochs=epochs,
        initial_epoch=startEpoch,
        validation_data=(x_val, y_val),
        callbacks=[checkPointCallback]
    )


def explicitTrain(
        model, preprocess_input,
        humanDataset: SegmentationDataset,
        nonHumanDataset: SegmentationDataset,
        valHumanDataset: SegmentationDataset,
        valNonHumanDataset: SegmentationDataset,
        trainingDir: str,
        modelEncoder: str,
        imageSize=224,
        batchSize: int = 1,
        epochs: int = 1,
        startEpoch: int = 0
):

    validationPacketSize = 16 * 16
    x_val_h, y_val_h = valHumanDataset.readBatch(validationPacketSize)
    augmentations.applyTransforms(x_val_h, y_val_h, augmentations.valid_transforms(imageSize))
    x_val_nh, y_val_nh = valNonHumanDataset.readBatch(validationPacketSize)
    augmentations.applyTransforms(x_val_nh, y_val_nh, augmentations.valid_transforms(imageSize))
    x_val = np.stack(x_val_h + x_val_nh)
    y_val = np.stack(y_val_h + y_val_nh)
    x_val = preprocess_input(x_val)

    # checkPointPath = os.path.join(trainingDir, 'u-net-{}.chpt'.format(modelEncoder))
    # checkPointCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkPointPath,
    #                                             save_weights_only=True,
    #                                             verbose=1)
    SAVE_AFTER_NUMBER = 50000
    packetSize = 16 * 16
    nonHumanPacketSize = max((packetSize * len(nonHumanDataset)) // len(humanDataset), 1)
    csv_logger = CSVLogger('training.log', append=True)

    for epoch in range(startEpoch, epochs):
        logger.info('epoch %d', epoch)
        humanDataset.reset()
        nonHumanDataset.reset()

        try:
            packets = len(humanDataset) // packetSize
            for packetIndex in range(packets - 1):
                logger.debug('reading batch, memory used %f', usedMemory())
                x_train_h, y_train_h = humanDataset.readBatch(packetSize)
                x_train_h, y_train_h = augmentations.appendTransforms(x_train_h, y_train_h,
                                                                      augmentations.train_transforms_after_resize,
                                                                      augmentations.resize_transforms(imageSize))
                logger.debug('reading human batch, memory used %f', usedMemory())
                x_train_nh, y_train_nh = nonHumanDataset.readBatch(nonHumanPacketSize)

                x_train_nh, y_train_nh = augmentations.appendTransforms(x_train_nh, y_train_nh,
                                                                        augmentations.train_transforms_after_resize,
                                                                        augmentations.resize_transforms(imageSize))
                logger.debug('reading nonHuman batch, memory used %f', usedMemory())
                x_train, y_train = HumanDatasetSequence.shuffleHumanNonHuman(x_train_h, x_train_nh, y_train_h,
                                                                             y_train_nh)
                x_train = np.concatenate((x_train,))
                y_train = np.concatenate((y_train,))
                del x_train_h
                del x_train_nh
                del y_train_h
                del y_train_nh
                logger.debug('concatenate batches, memory used %f', usedMemory())
                x_train = preprocess_input(x_train)
                # x_train = x_train / 255
                logger.debug('preprocess x_train, memory used %f', usedMemory())

                saveModel = ((humanDataset.index + nonHumanDataset.index) % SAVE_AFTER_NUMBER) < (
                            packetSize + nonHumanPacketSize)

                logger.debug('start train on %d samples, memory used %f', len(x_train), usedMemory())
                model.fit(
                    x=x_train,
                    y=y_train,
                    batch_size=batchSize,
                    epochs=epoch + 1,
                    initial_epoch=epoch,
                    validation_data=(x_val, y_val),
                    callbacks=[csv_logger]
                )
                if saveModel:
                    save_model(model, trainingDir, modelEncoder, packetIndex)
                del x_train
                del y_train
                logger.debug('trained on %d samples, memory used %f', humanDataset.index + nonHumanDataset.index,
                             usedMemory())
                gc.collect()
                # objgraph.show_most_common_types(limit=50)
                # obj = objgraph.by_type('list')[1000]
                # objgraph.show_backrefs(obj, max_depth=10)

            x_train_h, y_train_h = humanDataset.readBatch(packetSize)
            x_train_h, y_train_h = augmentations.appendTransforms(x_train_h, y_train_h,
                                                                  augmentations.train_transforms_after_resize,
                                                                  augmentations.resize_transforms(imageSize))
            x_train_nh, y_train_nh = nonHumanDataset.readBatch(nonHumanPacketSize)
            x_train_nh, y_train_nh = augmentations.appendTransforms(x_train_nh, y_train_nh,
                                                                    augmentations.train_transforms_after_resize,
                                                                    augmentations.resize_transforms(imageSize))
            x_train, y_train = HumanDatasetSequence.shuffleHumanNonHuman(x_train_h, x_train_nh, y_train_h, y_train_nh)
            x_train = np.concatenate((x_train,))
            y_train = np.concatenate((y_train,))
            del x_train_h
            del x_train_nh
            del y_train_h
            del y_train_nh
            x_train = preprocess_input(x_train)
            # x_train = x_train / 255

            model.fit(
                x=x_train,
                y=y_train,
                batch_size=batchSize,
                epochs=1,
                validation_data=(x_val, y_val),
                callbacks=[csv_logger]
            )
            save_model(model, trainingDir, modelEncoder, packets - 1)

            del x_train
            del y_train
            logger.info('epoch %d is trained', epoch)
        except Exception as e:
            logger.error('Exception %s', str(e))
            traceback.print_exc()
            return

        now = datetime.now()
        dt_string = now.strftime("%Y/%m/%d %H:%M:%S")

        modelPath = os.path.join(trainingDir, 'u-net-{}_epoch{}_{}.tfmodel'.format(modelEncoder, epoch, dt_string))
        model.save(modelPath)
        logger.info('model saved')


def loadCocoDataset(cocoAnnotations, datasetDir: str, classes: List = None, nonClasses: List = None,
                    shuffle: bool = False):
    datasetBuilder = CocoDatasetBuilder(datasetDir, cocoAnnotations=cocoAnnotations)
    if nonClasses is not None:
        datasetBuilder.selectAll().filterNonClasses(nonClasses)
    if classes is not None:
        datasetBuilder.addClasses(classes)
    dataset = datasetBuilder.build(shuffle)
    del datasetBuilder
    gc.collect()
    return dataset


def openSegmentationDatasets(datasetDir: str, train_h_number=61600, train_nh_number=28320,
                             val_h_number=2693, val_nh_number=2259):
    humanDataset = SegmentationDataset(os.path.join(datasetDir, 'human'), train_h_number, shuffle=True)
    nonHumanDataset = SegmentationDataset(os.path.join(datasetDir, 'nonHuman'), train_nh_number, shuffle=True)
    valHumanDataset = SegmentationDataset(os.path.join(datasetDir, 'valHuman'), val_h_number, shuffle=True)
    valNonHumanDataset = SegmentationDataset(os.path.join(datasetDir, 'valNonHuman'), val_nh_number, shuffle=True)
    return humanDataset, nonHumanDataset, valHumanDataset, valNonHumanDataset


def main():
    init_logging('training.log')

    logger.debug('gc enabled: {}'.format(gc.isenabled()))
    # gc.set_debug(gc.DEBUG_LEAK)

    args = parse_args()
    datasetDir = args.datasetDir
    trainingDir = args.trainingDir if args.trainingDir is not None else datasetDir
    learningRate = args.learningRate
    checkpointFilePath = args.checkpointFilePath
    modelEncoder = args.modelEncoder

    # efficientNet sizes: 224, 240, 260, 300, 380, 456, 528, 600
    # efficientNet optimizer: RMSprop, decay 0.9 and momentum 0.9; batch norm momentum 0.99, weight decay 1e-5; initial learning rate 0.256 that decays by 0.97 every 2.4 epochs
    # efficientNet special strategy: stochastic depth with survival probability 0.8, model size dependant dropout
    # if modelEncoder == 'resnet18':
    # optimizer = tfa.keras.optimizers.AdamW(
    optimizer = tf.keras.optimizers.Adam(
            lr=learningRate,
            # beta_1=0.9,
            # beta_2=0.999,
            # epsilon=1e-07,
            # amsgrad=False,
            # name='Adam'
        )
    # else: #modelEncoder == 'efficientNet3':
    #     optimizer = tf.keras.optimizers.RMSprop(
    #         lr=learningRate,
    #         momentum=0.9
    #     )

    model, preprocess_input = makeModel(optimizer, modelEncoder)
    if checkpointFilePath is not None:
        model.load_weights(checkpointFilePath)
        logger.info('model weights from %s are loaded', checkpointFilePath)

    humanDataset, nonHumanDataset, valHumanDataset, valNonHumanDataset = openSegmentationDatasets(datasetDir)
                                                                                            #,train_nh_number=0)
    explicitTrain(model, preprocess_input, humanDataset, nonHumanDataset, valHumanDataset, valNonHumanDataset,
                    trainingDir, modelEncoder, imageSize=320, batchSize=args.batchSize,
                    epochs=args.epochs, startEpoch=args.startEpoch
                  )


if __name__ == '__main__':
    main()
