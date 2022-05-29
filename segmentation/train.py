import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import copy

from unet.unet_model import UNet
from data import ImageDataset

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def train_model(model, train_loader, val_loader, criterion, epochs, optimizer):
    model = model.cuda()
    best_model = copy.deepcopy(model)
    train_losses = []
    val_losses = []
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        for x,y in train_loader:
            optimizer.zero_grad()
            x, y = x.cuda().float(), y.cuda().float()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        model.eval()
        no_corrects = 0
        total = 0
        val_loss = 0
        total_pixels = 0
        for x,y in val_loader:
            with torch.no_grad():
                x, y = x.cuda().float(), y.cuda().float()
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.shape[0]
                preds = torch.round(torch.sigmoid(logits))
                no_corrects += torch.sum(preds == y).item()
                total_pixels += torch.prod(torch.tensor(y.shape)).item()
                total += x.shape[0]
                
        val_acc = no_corrects / total_pixels
        print(f"Epoch {epoch+1}, validation accuracy: {val_acc}")
        val_losses.append(val_loss / total)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)
    
    return best_model, train_losses, val_losses

def plot_losses(train_losses, val_losses):
    plt.figure()
    n_epochs = len(val_losses)
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, val_losses, label='val loss')
    plt.legend()
    plt.title('Training Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.tight_layout()
    plt.savefig('segmentation_plot.png')

if __name__ == "__main__":
    model = UNet(1,1)

    train_dataset = ImageDataset("dataset/TRAIN", "dataset/TRAIN_anno")
    val_dataset = ImageDataset("dataset/VAL", "dataset/VAL_anno")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10
    best_model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, epochs, optimizer)
    plot_losses(train_losses, val_losses)
    torch.save(best_model, "best_model.pt")
    
    
    



