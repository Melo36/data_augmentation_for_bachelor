import cv2
import albumentations as A
import numpy as np
from utils import *
from PIL import Image
import time

ORIGINAL_IMAGES = 15
IMAGE_NAME = 'product'


affine_transformations_list = [
    A.Rotate((45,45), p=1),
    A.Rotate((315,315), p=1),
    A.HorizontalFlip(p=1),
    # nach oben
    A.ShiftScaleRotate(shift_limit_x=(0, 0), shift_limit_y=(0.2, 0.2),
                        scale_limit=(0, 0),
                        rotate_limit=(0, 0), p=1, border_mode=cv2.BORDER_DEFAULT),
    # nach unten
    A.ShiftScaleRotate(shift_limit_x=(0, 0), shift_limit_y=(-0.2, -0.2),
                        scale_limit=(0, 0),
                        rotate_limit=(0, 0), p=1, border_mode=cv2.BORDER_DEFAULT),
    # nach rechts
    A.ShiftScaleRotate(shift_limit_x=(0.25, 0.25), shift_limit_y=(0, 0),
                        scale_limit=(0, 0),
                        rotate_limit=(0, 0), p=1, border_mode=cv2.BORDER_DEFAULT),
    # nach links
    A.ShiftScaleRotate(shift_limit_x=(-0.25, -0.25), shift_limit_y=(0, 0),
                        scale_limit=(0, 0),
                        rotate_limit=(0, 0), p=1, border_mode=cv2.BORDER_DEFAULT),
    # unten rechts
    A.ShiftScaleRotate(shift_limit_x=(0.25, 0.25), shift_limit_y=(0.2, 0.2),
                        scale_limit=(0, 0),
                        rotate_limit=(0, 0), p=1, border_mode=cv2.BORDER_DEFAULT),
    # unten links
    A.ShiftScaleRotate(shift_limit_x=(-0.25, -0.25), shift_limit_y=(0.2, 0.2),
                        scale_limit=(0, 0),
                        rotate_limit=(0, 0), p=1, border_mode=cv2.BORDER_DEFAULT),
    # oben rechts
    A.ShiftScaleRotate(shift_limit_x=(0.25, 0.25), shift_limit_y=(-0.2, -0.2),
                        scale_limit=(0, 0),
                        rotate_limit=(0, 0), p=1, border_mode=cv2.BORDER_DEFAULT),
    # oben links
    A.ShiftScaleRotate(shift_limit_x=(-0.25, -0.25), shift_limit_y=(-0.2, -0.2),
                        scale_limit=(0, 0),
                        rotate_limit=(0, 0), p=1, border_mode=cv2.BORDER_DEFAULT),
    # scale to make smaller
    A.ShiftScaleRotate(shift_limit_x=(0, 0), shift_limit_y=(0,0),
                        scale_limit=(-0.5, -0.5),
                        rotate_limit=(0, 0), p=1, border_mode=cv2.BORDER_REPLICATE)
]

# scale to make bigger
'''A.ShiftScaleRotate(shift_limit_x=(0, 0), shift_limit_y=(0,0),
                        scale_limit=(0.5, 0.5),
                        rotate_limit=(0, 0), p=1, border_mode=cv2.BORDER_DEFAULT),'''

affine_list_len = len(affine_transformations_list)
print('Länge meiner affine list ', affine_list_len)

color_transformations_list = [
    A.RandomBrightnessContrast((0.1, 0.1), contrast_limit=0, p=1.0),
    #A.RandomBrightnessContrast((0.2, 0.2), contrast_limit=0, p=1.0),
    A.RandomBrightnessContrast((0.3, 0.3), contrast_limit=0, p=1.0),
    A.RandomBrightnessContrast((-0.1, -0.1), contrast_limit=0, p=1.0),
    #A.RandomBrightnessContrast((-0.2, -0.2), contrast_limit=0, p=1.0),
    A.RandomBrightnessContrast((-0.3, -0.3), contrast_limit=0, p=1.0),
    #A.Blur(blur_limit=(10, 10), p=1),
    #A.Blur(blur_limit=(20, 20), p=1)
]

# 1. resize images to 1080x1440
'''resize_transform = A.Resize(height=1440, width=1080, p=1.0)
for i in range(ORIGINAL_IMAGES):
    perform_transformation(resize_transform, f'img_{i}', i)
'''

# 2. Perform Rotation, Translation and Scaling
index = ORIGINAL_IMAGES

for i in range(ORIGINAL_IMAGES):
    for transform in affine_transformations_list:
        perform_transformation(transform, f'{IMAGE_NAME}_{i}', index)
        index += 1


number_images = ORIGINAL_IMAGES + affine_list_len * ORIGINAL_IMAGES

for i in range(number_images):
    for transform in color_transformations_list:
        perform_transformation(transform, f'{IMAGE_NAME}_{i}', index)
        index += 1


