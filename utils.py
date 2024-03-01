import random
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import albumentations as A
from PIL import Image


def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def plot_example(image, bbox, index):
    img_bbox = image.copy()
    img_bbox = visualize_bbox(img_bbox, bbox, 'haferflocken_ja')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'augmented_images/images/img_{index}.jpg', img)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #fig = plt.figure(figsize=(5, 5))
    #plt.imshow(img)
    #fig.add_subplot(2, 2, 4)
    plt.imshow(img_bbox)
    plt.show()


# From https://albumentations.ai/docs/examples/example_bboxes/
def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=10):
    """Visualizes a single bounding box on the image"""
    img_width = float(img.shape[1])
    img_height = float(img.shape[0])

    x_center, y_center, width, height = map(float, bbox)
    x_min = x_center - width / 2.0
    y_min = y_center - height / 2.0
    x_max = x_center + width / 2.0
    y_max = y_center + height / 2.0

    x_min *= img_width
    y_min *= img_height
    x_max *= img_width
    y_max *= img_height

    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    return img



def open_image_and_bbox(name):
    image = Image.open(f'augmented_images/images/{name}.jpg')

    file = open(f'augmented_images/labels/{name}.txt')
    if file:
        # read txt file to get bbox
        bbox = file.read().replace('\n', '').split(' ')
        class_type = bbox[0]
        # convert bbox from string to float
        bbox_float = [0.0, 0.0, 0.0, 0.0]
        for i in range(len(bbox) - 1):
            bbox_float[i] = float(bbox[i + 1])

        return image, bbox_float

    return image


def save_bbox(name, bbox, class_type):
    bbox = list(bbox)
    for i in range(len(bbox)):
        bbox[i] = round(bbox[i], 5)

    f = open(f'augmented_images/labels/{name}.txt', 'w')
    f.write(f'{class_type} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}')


def perform_transformation(transforms, name, index='test'):
    image, bbox_float = open_image_and_bbox(name)
    transform = A.Compose(
        [transforms], bbox_params=A.BboxParams(format="yolo", label_fields=[])
    )

    bboxes = [bbox_float]

    # transpose to get correct dimensions again
    #image = image.transpose(method=Image.Transpose.TRANSPOSE)
    #image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
    image = np.array(image)

    augmentations = transform(image=image, bboxes=bboxes)
    augmented_img = augmentations['image']
    augmented_bbox = augmentations['bboxes'][0]

    save_bbox(f'img_{index}', augmented_bbox, 0)
    plot_example(augmented_img, augmented_bbox, index)




