import argparse
import os
import tensorflow as tf
import logging

from broccole.model import makeModel
from broccole.logUtils import init_logging

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='save model')
    parser.add_argument('--checkpointFilePath', help='path to checkpoint', type=str, default=None)
    parser.add_argument('--modelFilePath', help='path to load/save model', type=str)
    args = parser.parse_args()
    return args

def main():
    init_logging('saveModel.log')

    args = parse_args()
    checkpointFilePath = args.checkpointFilePath
    modelFilePath = args.modelFilePath

    model, preprocess_input = makeModel()
    model.load_weights(checkpointFilePath)
    logger.info('model weights from %s are loaded', checkpointFilePath)
    model.save(modelFilePath)
    logger.info('model saved to %s', modelFilePath)

if __name__ == '__main__':
    main()
