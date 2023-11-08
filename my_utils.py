import os
import random

import numpy as np

import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch
from torchvision import transforms
import transforms as T
from tqdm.auto import tqdm


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


def calc_area(boxes):
    return (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    

def custom_collate(data):
    return data


class AFODataset(Dataset):
    def __init__(self, mode='train', transform=None, classes=None, 
                 root_img_path="/content/drive/MyDrive/FinalProject-CS321.O11/Dataset/images/", 
                 root_ann_path="/content/drive/MyDrive/FinalProject-CS321.O11/Dataset/labels/"):
        self.mode = mode
        self.transform = transform
        self.classes = classes
        self.aug_transform = T.Compose(
            [T.RandomHorizontalFlip(),
            T.RandomZoomOut(),
            T.RandomIoUCrop()]
        )
        # self.aug_transform = None
        
        self.img_path = os.path.join(root_img_path, mode)
        self.ann_path = os.path.join(root_ann_path, mode)

        print("Create dataset:")
        print(self.img_path)
        print(self.ann_path)
        
        self.img_names = os.listdir(self.img_path)
        self.img_names.sort()
        self._remove_background()
        
        if mode == 'train':
            self._augment()

    def _augment(self):
        def hsv(img):
            img_float32 = np.float32(img)
            # Convert the RGB image to HSV
            hsv_image = cv2.cvtColor(img_float32, cv2.COLOR_RGB2HSV)
            # Get a random delta between -100 and 100
            delta = np.random.randint(low=-100, high=100, size=1)[0]
            # Modify the hue by that quantity and convert it back
            hsv_image[:,:,0] = np.mod(hsv_image[:,:,0] + delta, 360.)
            rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

            return rgb_image

        def horiz_flip(img):
            img_float32 = np.float32(img)
            flip_img = cv2.flip(img_float32, 0) 
            
            return flip_img

        print("Augmentation")
        img_names_copy = self.img_names.copy()
        for img_name in img_names_copy: 
            img_file = os.path.join(self.img_path, img_name)
            
            img = cv2.imread(img_file)
            img_aug_hsv = hsv(img)
            # img_horiz_flip = horiz_flip(img)
            
            cv2.imwrite(img_file.replace(".jpg", "_aug_hsv.jpg"), img_aug_hsv)
            # cv2.imwrite(img_horiz_flip, img_file.replace(".jpg", "_horiz_flip.jpg")) 
            
            self.img_names.append(img_name.replace(".jpg", "_aug_hsv.jpg"))
            # self.img_names.append(img_file.replace(".jpg", "_horiz_flip.jpg"))
        
        print(len(img_names_copy), "--", len(self.img_names))
        
    # Remove background images
    def _remove_background(self, verbose=False):
        _count = 0
        img_names_copy = self.img_names.copy()
        progress_bar = tqdm(range(len(self.img_names)))

        for img_name in img_names_copy:
            ann_path = os.path.join(self.ann_path, img_name.replace("jpg", "txt"))
            with open(ann_path, 'r') as f:
                annot = [line.strip() for line in f.readlines()]

            annot = np.array(str2float_annot(annot))
            if annot.size == 0:
                if verbose:
                    print(f"{img_name} is treated as background image!")

                self.img_names.remove(img_name)
                _count = _count + 1
                
            progress_bar.update(1) 

        print(f"Total removed background images: {_count}")
        print(f"{len(img_names_copy)} -- {len(self.img_names)}")
        progress_bar.close()

    def _load_images(self, img_name):
        img = Image.open(os.path.join(self.img_path, img_name)).convert('RGB')

        if self.transform:
            img = self.transform(img)  # TODO: img.size: width x height

        return img

    def _load_annot(self, img_name, img_width=None, img_height=None):
        flip = False
        if "_aug_hsv" in img_name:
            img_name = img_name.replace("_aug_hsv", "")
        
        ann_path = os.path.join(self.ann_path, img_name[:-3] + 'txt')
    
        with open(ann_path, 'r') as f:
            annot = [line.strip() for line in f.readlines()]

        # [ cls, x, y, w h ]
        annot = np.array(str2float_annot(annot))

        if annot.size == 0:
            print(f"{img_name} is treated as background image!")
            annot = np.zeros((1, 5))
            annot[0][0] = -1

        labels = annot[:, 0].astype(int) + 1  # 0s for background class
        boxes = annot[:, 1:]
        boxes = xywh2xyxy(boxes, img_width=img_width, img_height=img_height)
        
        # if flip:
        return torch.from_numpy(boxes), torch.from_numpy(labels)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img = self._load_images(img_name)
        channel, height, width = img.shape  # TODO: img.size if not transform

        target = {}
        target["boxes"], target["labels"] = self._load_annot(img_name, img_width=width, img_height=height)
        target["image_id"] = img_name.replace("txt", '')
        target["area"] = calc_area(target["boxes"])
        # suppose all instances are not crowd
        target["iscrowd"] = torch.zeros((len(target["boxes"]), ), dtype=torch.int64)

        if self.aug_transform is not None:
            img, target = self.aug_transform(img, target)
        
        return img, target
