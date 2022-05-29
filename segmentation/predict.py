import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import cv2
import skimage.transform as trans
import sys
import os
from unet.unet_model import UNet
from data import ImageDataset

model = torch.load("best_model.pt")

if __name__ == "__main__":
    if not os.path.isdir("test_out/"):
        os.mkdir("test_out/")
    
    for i in range(701,801):
        image_path = f"dataset/TEST/y{i}.jpg"
        anno_path = f"dataset/TEST_anno/y{i}.png"
        anno = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image / 255
        image = trans.resize(image,(256,256))


        org = cv2.imread(image_path)
        rows, cols, channels = org.shape     
        input = torch.tensor(image).cuda().float().unsqueeze(0).unsqueeze(0)
        predicted = torch.round(torch.sigmoid(model(input)))
        predicted = predicted.detach().squeeze().cpu().numpy()

        predicted = predicted.astype(np.float64) * 255
        predicted = trans.resize(predicted, (rows,cols))
        predicted = predicted.astype(np.uint8)
        predicted = np.expand_dims(predicted, axis=2)
        predicted = np.repeat(predicted, 3, axis=2)  
        ret, mask = cv2.threshold(predicted, 0, 255, cv2.THRESH_BINARY)
        white_pixels = np.where((mask[:, :, 0] == 255) & 
                                (mask[:, :, 1] == 255) & 
                                (mask[:, :, 2] == 255))
        mask[white_pixels] = [0, 0, 255]
        add = cv2.addWeighted(org, 0.9, mask, 0.7, 0)

        anno = np.expand_dims(anno, axis=2)
        anno = np.repeat(anno, 3, axis=2)   
        ret2, mask2 = cv2.threshold(anno, 0, 255, cv2.THRESH_BINARY)
        white_pixels2 = np.where((mask2[:, :, 0] == 255) & 
                                (mask2[:, :, 1] == 255) & 
                                (mask2[:, :, 2] == 255))
        mask2[white_pixels2] = [0, 0, 255]
        add2 = cv2.addWeighted(org, 0.9, mask2, 0.7, 0)


        add = np.concatenate([add, add2], axis=1)
        cv2.imwrite(f'test_out/y{i}.png',add)

