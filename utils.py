import os
import random

import numpy as np

import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt



def loadImages(images_list, num):
    loaded_images = []

    for i in range(num):
        idx = random.randint(0, len(images_list))
        img = cv2.imread(images_list[idx])
        loaded_images.append(img)

    return loaded_images


def loadImgAnn(images_path, annotations_path, num):
    images_list = os.listdir(images_path)
    annotations_list = os.listdir(annotations_path)

    loaded_images = []
    loaded_annotations = []

    for i in range(num):
        idx = random.randint(1, len(images_list))
        print(idx)
        img_path = os.path.join(images_path, images_list[idx])

        if not os.path.isFile(img_path):
            continue

        img = Image.open(img_path)
        loaded_images.append(img)

        ann_path = os.path.join(annotations_path, img_path[len(images_path):-3] + 'txt')  # TODO: check images_path
        with open(ann_path, 'r') as f:
            data = [line for line in f.readlines()]
            loaded_annotations.append(data)

    return loaded_images, loaded_annotations


def floatAnnotation(annotations_list):
    coords = []
    float_coords = []

    for i in annotations_list:
        temp = []

        for ii in i:
            temp.append(ii.split())

        coords.append(temp)

    for i in coords:
        fltemp = []

        for ii in i:
            if len(ii) > 0:
                z = [float(iii) for iii in ii]
                fltemp.append(z)

        float_coords.append(fltemp)

    return float_coords


def imgCoords(coords, img):
    img_size = img.size
    img_width = img_size[0]
    img_height = img_size[1]

    x1 = float(img_width) * (2.0 * float(coords[1]) - float(coords[3])) / 2.0
    y1 = float(img_height) * (2.0 * float(coords[2]) - float(coords[4])) / 2.0
    x2 = float(img_width) * (2.0 * float(coords[1]) + float(coords[3])) / 2.0
    y2 = float(img_height) * (2.0 * float(coords[2]) + float(coords[4])) / 2.0

    return [int(x1), int(y1), int(x2), int(y2)]


def drawAnnotation(img, annotList):
#     npimg = np.array(img)
#     cvImage = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
    img_copy = img.copy()
    for annot in annotList:
        drawImg = ImageDraw.Draw(img_copy)
        drawImg.rectangle(annot, outline=(255, 0, 0), width=4)
#         cv2.rectangle(cvImage, (annot[0], annot[1]), (annot[2], annot[3]), (255,0,0), 2)
    plt.imshow(img_copy)
    plt.axis('off')


def xywh2xyxy(boxes_xywh, img_width, img_height):
    boxes_xyxy = []

    for box in boxes_xywh:
        x, y, w, h = box

        x1 = int(float(img_width) * (2.0 * float(x) - float(w)) / 2.0)
        y1 = int(float(img_height) * (2.0 * float(y) - float(h)) / 2.0)
        x2 = int(float(img_width) * (2.0 * float(x) + float(w)) / 2.0)
        y2 = int(float(img_height) * (2.0 * float(y) + float(h)) / 2.0)

        boxes_xyxy.append([x1, y1, x2, y2])

    return np.array(boxes_xyxy)


def str2float_annot(annotations_list):
    coords = []
    float_coords = []

    for i in annotations_list:
        coords.append(i.split())

    for i in coords:
        temp = []
        for ii in i:
            temp.append(float(ii))

        float_coords.append(temp)

    return float_coords