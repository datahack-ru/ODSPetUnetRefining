import argparse
import cv2
import numpy as np
import onnxruntime as rt
import onnx
import logging

from broccole.Timer import Timer
from broccole.logUtils import init_logging
import broccole.augmentations as augmentations

logger = logging.getLogger(__name__)


def inference(modelFilePath: str):
    imageSize = 224

    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = rt.InferenceSession(modelFilePath, sess_options)
    input_name = sess.get_inputs()[0].name
    print("input_name {}".format(input_name))
    mask_name = sess.get_outputs()[0].name
    print("mask_name {}".format(mask_name))

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

        masks = sess.run(None, {input_name: imgs})[0]

        inferenceTime.end()

        for i in range(masks.shape[0]):
            image = imgs[i]
            mask = masks[i]
            threshold = np.copy(mask)
            maskThreshold = 0.3  # 150 / 255
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
    parser.add_argument('--modelFilePath', help='path to model', type=str)
    args = parser.parse_args()
    return args


def main():
    init_logging('webcamInferenceOnnx.log')

    args = parse_args()
    modelFilePath = args.modelFilePath

    inference(modelFilePath)


if __name__ == '__main__':
    main()
