import json
import cv2
import os
import random
import numpy as np
from typing import List
from pycocotools import mask as cocoMask
import logging

logger = logging.getLogger(__name__)

from broccole.CocoDataset import CocoDataset

class CocoDatasetBuilder:
    def __init__(self, imagesDir: str, annotationFilePath: str = None, cocoAnnotations = None):
        if cocoAnnotations is not None:
            self.cocoAnnotations = cocoAnnotations
        else:
            with open(annotationFilePath, 'r') as annotationFile:
                self.cocoAnnotations = json.load(annotationFile)
                logger.debug("coco annotations are loaded")
        self.imagesDir = imagesDir

        self.annotations = {}

    def __del__(self):
        del self.annotations
        del self.cocoAnnotations

    @staticmethod
    def __annotationInfo__(annotation):
        return {
            "iscrowd": annotation["iscrowd"],
            "segmentation": annotation["segmentation"],
            "category_id": annotation["category_id"]
        }

    def selectAll(self):
        self.annotations = {}
        for annotation in self.cocoAnnotations["annotations"]:
            id = annotation["image_id"]
            if id in self.annotations:
                imageAnnotations = self.annotations[id]
            else:
                imageAnnotations = []
                self.annotations[id] = imageAnnotations
            imageAnnotations.append(CocoDatasetBuilder.__annotationInfo__(annotation))
        logger.debug("all annotations are selected")
        return self

    def addClasses(self, classes: List):
        if len(classes) == 0:
            raise Exception('classes is empty list')
        classes = { cls: cls for cls in classes }

        for annotation in self.cocoAnnotations["annotations"]:
            if annotation["category_id"] in classes:
                id = annotation["image_id"]
                if id in self.annotations:
                    imageAnnotations = self.annotations[id]
                else:
                    imageAnnotations = []
                    self.annotations[id] = imageAnnotations
                imageAnnotations.append(CocoDatasetBuilder.__annotationInfo__(annotation))
        logger.debug("classes %s annotations are added", str(classes))
        return self

    def filterNonClasses(self, classes: List):
        if len(classes) == 0:
            raise Exception('classes is empty list')
        classes = { cls: cls for cls in classes }

        idsToDelete = []

        for id, imageAnnotations in self.annotations.items():
            hasClass = False
            for annotation in imageAnnotations:
                if annotation["category_id"] in classes:
                    hasClass = True
                    break
            if hasClass:
                idsToDelete.append(id)
        for id in idsToDelete:
            self.annotations.pop(id)
        del idsToDelete

        logger.debug("not classes %s annotations are filtered", str(classes))
        return self

    def addNonClasses(self, classes: List, maxCount: int = None, shuffle: bool = False):
        if len(classes) == 0:
            raise Exception('classes is empty list')
        classes = { cls: cls for cls in classes }

        indices = list(range(len(self.cocoAnnotations["annotations"])))
        if shuffle:
            random.shuffle(indices)

        count = 0
        for index in indices:
            annotation = self.cocoAnnotations["annotations"][index]
            if maxCount is not None and count >= maxCount:
                break

            if annotation["category_id"] not in classes:
                id = annotation["image_id"]
                if id in self.annotations:
                    imageAnnotations = self.annotations[id]
                else:
                    imageAnnotations = []
                    self.annotations[id] = imageAnnotations
                imageAnnotations.append(CocoDatasetBuilder.__annotationInfo__(annotation))
                count += 1
        logger.debug("not classes %s annotations are added", str(classes))
        return self

    def build(self, shuffle: bool = False):
        images =  [
            {
                "id": image["id"],
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
            }
            for image in self.cocoAnnotations["images"] if image["id"] in self.annotations
        ]

        dataset = CocoDataset(self.annotations, images, self.imagesDir, shuffle)
        logger.debug('dataset is built')
        return dataset

    def __len__(self):
        return len(self.annotations)
