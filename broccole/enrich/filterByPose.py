import os
import argparse
import cv2
import numpy as np
import logging

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

FORMAT = "[%(asctime)-15s] %(levelname)s  %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def parseArgs():
    parser = argparse.ArgumentParser(description='train U-Net')
    parser.add_argument('--imagesDir', help='path to directory with images', type=str)
    parser.add_argument('--filteredDir', help='path to directory with filtered images', type=str)
    args = parser.parse_args()
    return args


def loadDetectorModel():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def loadKeypointsModel():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def main():
    args = parseArgs()
    imagesDir = args.imagesDir
    filteredDir = args.filteredDir

    detector, _ = loadDetectorModel()
    keypointsDetector, keypointsCfg = loadKeypointsModel()

    humanClass = 0
    for imageFileName in os.listdir(imagesDir):
        imagePath = os.path.join(imagesDir, imageFileName)
        if os.path.isfile(imagePath) and imageFileName.endswith(".jpg"):
            image = cv2.imread(imagePath)
            detected = detector(image)["instances"]
            humanCount = detected.pred_classes.tolist().count(humanClass)
            logger.debug("image {}".format(imageFileName))
            if 0 <= humanCount <= 5:
                hasHandsUp = False
                keypoints = keypointsDetector(image)["instances"]
                # "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
                for i in range(len(keypoints.scores)):
                    if keypoints.pred_classes[i] == humanClass:
                        joints = keypoints.pred_keypoints[i]
                        leftPalmJoint = joints[9]  # left_wrist
                        rightPalmJoint = joints[10]  # right_wrist
                        headJoint = joints[0]  # nose
                        if headJoint[2] > 0:
                            if (leftPalmJoint[2] > 0 and leftPalmJoint[1] < headJoint[1]) or (rightPalmJoint[2] > 0 and rightPalmJoint[1] < headJoint[1]):
                                # one or two hands are above head
                                hasHandsUp = True
                                break

                maskIndecies = []
                for i in range(len(detected.scores)):
                    if detected.pred_classes[i] == humanClass:
                        maskIndecies.append(i)
                if hasHandsUp and len(maskIndecies) > 0:
                    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(keypointsCfg.DATASETS.TRAIN[0]), scale=1.2)
                    out = v.draw_instance_predictions(keypoints.to("cpu"))
                    cv2.imshow("vis", out.get_image()[:, :, ::-1])
                    # cv2.imshow("image", image)
                    cv2.waitKey(1)

                    cv2.imwrite(os.path.join(filteredDir, imageFileName), image)
                    maskName = os.path.splitext(imageFileName)[0] + "_mask.png"
                    masks = detected.pred_masks.cpu().numpy()
                    masks = np.take(masks, maskIndecies, 0)
                    mask = np.amax(masks, 0)
                    logger.debug("mask {}".format(maskName))
                    cv2.imwrite(os.path.join(filteredDir, maskName), (mask * 255).astype(np.uint8))

if __name__ == "__main__":
    main()