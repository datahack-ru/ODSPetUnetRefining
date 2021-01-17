import os
import argparse
import logging
import cv2
import numpy as np

from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, calibrate, CalibrationDataReader
import onnxruntime as rt

from broccole.SegmentationDataset import SegmentationDataset
from broccole.logUtils import init_logging

logger = logging.getLogger(__name__)

class ModelCalibrationDataReader(CalibrationDataReader):
    def __init__(self, datasetDir: str, modelFilePath: str):
        self.datasetDir = datasetDir

        sess = rt.InferenceSession(modelFilePath)
        inputNode = sess.get_inputs()[0]
        self.input_name = inputNode.name
        # (_, height, width, _) = inputNode.shape

        self.imageSize = 224
        print('imageSize {}'.format(self.imageSize))
        self.humanDataset = SegmentationDataset(os.path.join(self.datasetDir, 'valHuman'))
        self.nonHumanDataset = SegmentationDataset(os.path.join(self.datasetDir, 'valNonHuman'))
        print('humanDataset size {}'.format(len(self.humanDataset)))
        print('nonHumanDataset size {}'.format(len(self.nonHumanDataset)))
        self.packetSize = 4 * 4
        self.nonHumanPacketSize = max((self.packetSize * len(self.nonHumanDataset)) // len(self.humanDataset), 1)
        self.datasize = \
            (len(self.humanDataset) // self.packetSize) * self.packetSize \
            + (len(self.nonHumanDataset) // self.nonHumanPacketSize) * self.nonHumanPacketSize

        data_dicts = [{ self.input_name: image } for image in self.representative_dataset_gen()]
        self.datasize = len(data_dicts)
        self.enum_data_dicts = iter(data_dicts)

    def get_next(self):
        return next(self.enum_data_dicts, None)
        # for image in self.representative_dataset_gen():
        #     yield { self.input_name: image }

    def representative_dataset_gen(self):
        packets = len(self.humanDataset) // self.packetSize
        for i in range(packets):
            x_train_h = self.humanDataset.readImageBatch(self.packetSize)
            x_train_nh = self.nonHumanDataset.readImageBatch(self.nonHumanPacketSize)
            x_train = x_train_h + x_train_nh
            print('packet {}'.format(i))
        
            for j in range(len(x_train)):
                resized = cv2.resize(x_train[j], (self.imageSize, self.imageSize))
                resized = resized.reshape(1, resized.shape[0], resized.shape[1], resized.shape[2])
                resized = resized.astype(np.float32)
                yield resized

def parse_args():
    parser = argparse.ArgumentParser(description='quantize onnx model')
    parser.add_argument('--datasetDir', help='dataset for calibration', type=str)
    parser.add_argument('--modelFilePath', help='path to onnx model', type=str)
    parser.add_argument('--quantizedModelFilePath', help='path to save quantized onnx model', type=str)
    args = parser.parse_args()
    return args

def main():
    init_logging('quantizeOnnx.log')

    args = parse_args()
    modelFilePath = args.modelFilePath
    quantizedModelFilePath = args.quantizedModelFilePath
    datasetDir = args.datasetDir

    # dr = ModelCalibrationDataReader(datasetDir, modelFilePath)
    # quantize_static(modelFilePath, quantizedModelFilePath, dr)
    quantize_dynamic(modelFilePath, quantizedModelFilePath, weight_type=QuantType.QInt8)
    print('Calibrated and quantized model saved.')


if __name__ == '__main__':
    main()
