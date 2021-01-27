import tensorflow as tf
import numpy as np
import logging
import gc
from datetime import date
from pathlib import Path
import shutil
import cv2

from broccole.SegmentationDataset import SegmentationDataset
from broccole.logUtils import init_logging
from broccole.trainPrepared import parse_args, openSegmentationDatasets
from broccole.model import makeModel

logger = logging.getLogger(__name__)


def iou_coef(y_true, y_pred, smooth=1):
    """
    Both y_true and y_pred are N x H x W, where N is a batch size, H is a height of images, W is a width of images.

    Ref: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    :param y_true: True masks
    :param y_pred: Predicted masks
    :param smooth: smoothness coefficient
    :return: IoU score
    """
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=[1, 2])
    union = tf.keras.backend.sum(y_true, [1, 2]) + tf.keras.backend.sum(y_pred, [1, 2]) - intersection
    iou = tf.keras.backend.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def save_images(indices, pred, data_dir):
    """
    Save the worst images and predicted masks to ./bad_predictions_{TIMESTAMP} folder.
    :param indices: Indices of bad images.
    :param pred: Predicted masks.
    :param data_dir: Data path.
    :return: None
    """
    folder = Path(data_dir, 'bad_predictions_', date.today().strftime('%Y-%m-%d_%H-%M-%S'))
    Path(folder).mkdir(exist_ok=True)  # create new folder for the images
    val_human_path = Path(data_dir, 'valHuman')
    val_images = list(val_human_path.glob('*.jpg'))  # get a list of all images in the folder
    for i in indices:
        img_name = val_images[i]
        shutil.move(Path(val_human_path, img_name), Path(folder))  # move image
        mask_name = 'mask' + img_name[:5]
        shutil.move(Path(val_human_path, mask_name), Path(folder))  # move mask
        pred_name = 'pred' + img_name[:5][:-4] + '.jpg'
        cv2.imwrite(Path(val_human_path, pred_name), pred[i, :, :].numpy())  # save predicted mask


def evaluate(model, val_human, data_dir, k=30):
    """
    Predict masks and save the worst ones.

    :param model: Model instance.
    :param val_human: Validation human data.
    :param data_dir: Data path.
    :param k: Number of images to save.
    :return: None
    """
    logging.debug('Evaluation started')
    predictions = []
    masks = []
    batch_size = 64
    limit = len(val_human) // batch_size

    for i in range(limit - 1):  # iterate over validation data
        imgs_, masks_ = val_human.readBatch(batch_size)
        pred = model.predict(imgs_)
        predictions.append(pred)
        masks.append(masks_)

    predictions_tf = tf.concat(predictions, axis=0)
    masks_tf = tf.concat(masks, axis=0)
    scores = iou_coef(masks_tf, predictions_tf)
    indices = tf.argsort(scores)[:k]  # indices of the worst predicted images
    save_images(indices, predictions_tf, data_dir)


def main():
    init_logging('active_learning.log')

    logger.debug('gc enabled: {}'.format(gc.isenabled()))
    # gc.set_debug(gc.DEBUG_LEAK)

    args = parse_args()
    datasetDir = args.datasetDir
    trainingDir = args.trainingDir if args.trainingDir is not None else datasetDir
    learningRate = args.learningRate
    checkpointFilePath = args.checkpointFilePath
    modelEncoder = args.modelEncoder

    # efficientNet sizes: 224, 240, 260, 300, 380, 456, 528, 600
    # efficientNet optimizer: RMSProp, decay 0.9 and momentum 0.9; batch norm momentum 0.99, weight decay 1e-5; initial learning rate 0.256 that decays by 0.97 every 2.4 epochs
    # efficientNet special strategy: stochastic depth with survival probability 0.8, model size dependant dropout
    optimizer = tf.keras.optimizers.Adam(
        lr=learningRate,
        # beta_1=0.9,
        # beta_2=0.999,
        # epsilon=1e-07,
        # amsgrad=False,
        # name='Adam'
    )

    model, preprocess_input = makeModel(optimizer, modelEncoder)
    if checkpointFilePath is not None:
        model.load_weights(checkpointFilePath)
        logger.info('model weights from %s are loaded', checkpointFilePath)

    humanDataset, nonHumanDataset, valHumanDataset, valNonHumanDataset = openSegmentationDatasets(datasetDir)
    evaluate(model, valHumanDataset, datasetDir)


if __name__ == '__main__':
    main()
