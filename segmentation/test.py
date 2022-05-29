import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
from torchmetrics import JaccardIndex

from unet.unet_model import UNet
from data import ImageDataset


def calculate_tp_fp(pred, y):
    pred_labels, true_labels = pred.long().flatten().detach().cpu().numpy(), y.long().flatten().detach().cpu().numpy()
    
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    return TP, TN, FP, FN
    

if __name__ == "__main__":
    
    model = torch.load("best_model.pt")
    test_dataset = ImageDataset("dataset/TEST", "dataset/TEST_anno")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    jaccard = JaccardIndex(num_classes=2)

    total = 0
    dices, ious, accs, precisions, recalls, sensitivities, specificities, f_scores = 0,0,0,0,0,0,0,0
    accs = 0
    for x,y in test_loader:
        with torch.no_grad():
            x, y = x.cuda().float(), y.cuda().float()
            B = x.shape[0]
            logits = model(x)
            preds = torch.round(torch.sigmoid(logits))
            
            TP, TN, FP, FN = calculate_tp_fp(preds, y)
            
            dice = (2 * TP) / (2 * TP + FP + FN)
            dices += dice * B
            
            iou = jaccard(preds.long().cpu(), y.long().cpu())
            ious += iou.item() * B
            
            acc = (TP + TN)/(TP + TN + FP + FN)
            accs += acc * B
            
            precision = TP / (TP + FP)
            precisions += precision * B
            
            recall = TP / (TP + FN)
            recalls += recall * B
            
            sensitivity = TP / (TP + FN)
            sensitivities += sensitivity * B
            
            specificity = TN / (TN + FP)
            specificities = specificity * B
            
            f_score = 2 * precision * recall / (precision + recall)
            f_scores = f_score * B
            
            total += B
    
    mean_precision = precisions / total
    mean_dice = dices / total
    mean_iou = ious / total
    test_acc = accs / total
    precisions = precisions / total
    recalls = recalls / total
    sensitivities = sensitivities / total
    specificities = specificities / total
    f_scores = f_scores / total
    
    
    print(f"Test set Accuracy: {test_acc}")
    print(f"Test set IoU: {mean_iou}")
    print(f"Test set Dice Coefficient: {mean_dice}")
    print(f"Test set Precision: {precisions}")
    print(f"Test set Recall: {recalls}")
    print(f"Test set Sensitivity: {sensitivities}")
    print(f"Test set Specificity: {specificities}")
    print(f"Test set F Score: {f_scores}")
