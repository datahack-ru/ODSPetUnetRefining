import albumentations as albu
import cv2
import numpy as np

def pre_transforms(image_size=224):
    return [albu.Resize(image_size, image_size, p=1)]

def hard_transforms():
    result = [
      albu.RandomRotate90(),
      albu.CoarseDropout(),# Cutout(),
      albu.RandomBrightnessContrast(
          brightness_limit=0.2, contrast_limit=0.2, p=0.3
      ),
      albu.GridDistortion(p=0.3),
      albu.HueSaturationValue(p=0.3)
    ]

    return result

def hard_transforms_2():
    black = (0, 0, 0)

    result = [
      albu.Flip(),
      albu.RandomRotate90(),
      albu.OneOf([
            albu.IAAAdditiveGaussianNoise(),
            albu.GaussNoise(),
        ], p=0.2),
      albu.CoarseDropout(),# Cutout(),
      albu.OneOf(
        [
        #   albu.MotionBlur(p=0.2),
          albu.MedianBlur(p=0.1),
          albu.Blur(blur_limit=3, p=0.1)
        ],
        p=0.2
      ),
      albu.ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.2, rotate_limit=45,
        border_mode=cv2.BORDER_CONSTANT, value=black, mask_value=black,
        p=0.2
      ),
      albu.OneOf(
        [
          albu.CLAHE(clip_limit=2),
          albu.IAASharpen(),
        #   IAAEmboss(),
          albu.RandomBrightnessContrast(
              brightness_limit=0.2, contrast_limit=0.2
          ),
        
        ],
        p=0.3
      ),
      albu.OneOf(
        [
          albu.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=black, mask_value=black, p=0.3),
          albu.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=black, mask_value=black, p=0.1),
          albu.IAAPiecewiseAffine(p=0.3),
        ],
        p=0.2
      ),
      albu.HueSaturationValue(p=0.3),
    ]

    return result
    

def resize_transforms(image_size=224):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.5)

    random_crop = albu.Compose([
      albu.SmallestMaxSize(pre_size, p=1),
      albu.RandomCrop(
          image_size, image_size, p=1
      )

    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose([
      albu.LongestMaxSize(pre_size, p=1),
      albu.RandomCrop(
          image_size, image_size, p=1
      )

    ])

    # Converts the image to a square of size image_size x image_size
    result = [
      albu.OneOf([
          random_crop,
          rescale,
          random_crop_big
      ], p=1)
    ]

    return result

def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize()]

def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
      item for sublist in transforms_to_compose for item in sublist
    ])
    return result

def train_transforms(image_size):
  return compose([
    resize_transforms(image_size), 
    hard_transforms_2(), 
    post_transforms()
  ])

post_only_transforms = compose([
    post_transforms()
])

train_transforms_after_resize = compose([
    hard_transforms_2(), 
    post_transforms()
])

def valid_transforms(image_size):
  return compose([
    pre_transforms(image_size),
    post_transforms()
  ])

def show_transforms(image_size: int):
  return compose([
    resize_transforms(image_size),
    hard_transforms_2()
])

def applyTransforms(images, masks, transforms):
    for i in range(len(images)):
        data = {
            'image': images[i],
            'mask': masks[i]
        }
        augData = transforms(**data)
        images[i] = augData['image'].astype(np.float32)
        masks[i] = augData['mask'].astype(np.float32)
    return images, masks

def appendTransforms(images, masks, transforms, resize_transforms = None, post_only_transforms = post_only_transforms):
    newImages = []
    newMasks = []

    for i in range(len(images)):
        data = {
            'image': images[i],
            'mask': masks[i]
        }
        if resize_transforms is not None:
            data = compose([resize_transforms])(**data)
            resizedData = post_only_transforms(**data)
            images[i] = resizedData['image'].astype(np.float32)
            masks[i] = resizedData['mask'].astype(np.float32)
        augData = transforms(**data)
        newImages.append(augData['image'].astype(np.float32))
        newMasks.append(augData['mask'].astype(np.float32))
    return images + newImages, masks + newMasks
