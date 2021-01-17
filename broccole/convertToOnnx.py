import argparse
import os
import tensorflow as tf
import logging
import subprocess

from broccole.model import makeModel
from broccole.logUtils import init_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='train U-Net')
    parser.add_argument('--checkpointFilePath', help='path to checkpoint', type=str, default=None)
    parser.add_argument('--modelFilePath', help='path to load/save model', type=str)
    parser.add_argument('--onnxFilePath', help='path to save onnx model', type=str)
    args = parser.parse_args()
    return args

def main():
    init_logging('tf2onnx.log')

    args = parse_args()
    checkpointFilePath = args.checkpointFilePath
    modelFilePath = args.modelFilePath
    onnxFilePath = args.onnxFilePath

    if checkpointFilePath is not None:
        model, preprocess_input = makeModel()
        model.load_weights(checkpointFilePath)
        logger.info('model weights from %s are loaded', checkpointFilePath)
        model.save(modelFilePath)
        logger.info('model saved to %s', modelFilePath)

    # python -m tf2onnx.convert --saved-model tensorflow-model-path --opset 10 --output model.onnx
    subprocess.run([
        "python3.8", "-m", "tf2onnx.convert",
        "--saved-model", modelFilePath, "--opset", "11", "--output", onnxFilePath
    ])

if __name__ == '__main__':
    main()
