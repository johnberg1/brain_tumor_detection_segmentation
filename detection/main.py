import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import copy

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def train_model(model, train_loader, val_loader, criterion, epochs, optimizer):
    model = model.cuda()
    best_model = copy.deepcopy(model)
    train_losses = []
    test_losses = []
    best_corrects = 0
    for epoch in range(epochs):
        model.train()
        for x,y in train_loader:
            optimizer.zero_grad()
            x,y = x.cuda(), y.cuda().float()
            scores = model(x)
            loss = criterion(scores.squeeze(1), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        no_corrects = 0
        total = 0
        test_loss = 0
        for x,y in val_loader:
            with torch.no_grad():
                x,y = x.cuda(), y.cuda().float()
                scores = model(x)
                loss = criterion(scores.squeeze(1), y)
                test_loss += loss.item() * x.shape[0]
                preds = torch.round(torch.sigmoid(scores)).squeeze(1)
                no_corrects += torch.sum(preds == y.data)
                total += len(y)
         
        test_losses.append(test_loss / total)
        acc = no_corrects/total
        print(f"Epoch {epoch+1}, validation accuracy: {acc}")
        if no_corrects > best_corrects:
            best_corrects = no_corrects
            best_model = copy.deepcopy(model)
        
    return best_model, train_losses, test_losses

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
    plt.savefig('detection_plot.png')

def test_model(model, test_loader):
    model.eval()
    model = model.cuda()
    no_corrects = 0
    total = 0
    predictions = []
    labels = []
    for x,y in test_loader:
        with torch.no_grad():
            x,y = x.cuda(), y.cuda().float()
            scores = model(x)
            preds = torch.round(torch.sigmoid(scores)).squeeze(1)
            predictions.extend(preds.tolist())
            labels.extend(y.data.tolist())


    acc = accuracy_score(predictions,labels)
    precision = precision_score(predictions,labels)
    recall = recall_score(predictions,labels)
    f1 = f1_score(predictions,labels)
    print("Test set metrics:")
    print(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256,256)),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = ImageFolder("dataset/", transform = transform)
    val_per = 0.1
    test_per = 0.2
    original_len = len(dataset)
    test_len = int(original_len * test_per)
    val_len = int(original_len * val_per)
    train_len = original_len - val_len - test_len
    train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len, val_len])
    
    train_loader = DataLoader(train_set, batch_size = 64, shuffle = True, num_workers = 2)
    test_loader = DataLoader(test_set, batch_size = 64, shuffle = False, num_workers = 2)
    val_loader = DataLoader(val_set, batch_size = 64, shuffle = False, num_workers = 2)
    
    model = nn.Sequential(nn.Conv2d(3, 32, 5, stride=1, padding=2, bias=True), # B x 32 x 256 x 256
                      nn.ReLU(),
                      nn.MaxPool2d(2), # B x 32 x 128 x 128
                      nn.Dropout(0.5),
                      nn.Conv2d(32, 32, 5, stride=1, padding=2, bias=True), # B x 32 x 128 x 128
                      nn.ReLU(),
                      nn.MaxPool2d(2), # B x 32 x 64 x 64
                      nn.Dropout(0.3),
                      nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=True), # B x 64 x 64 x 64
                      nn.ReLU(),
                      nn.MaxPool2d(2), # B x 64 x 32 x 32
                      nn.Dropout(0.3),
                      nn.Flatten(), 
                      nn.Linear(64 * 32 * 32, 128), # B x 128
                      nn.ReLU(),
                      nn.Dropout(0.15),
                      nn.Linear(128, 1)) # B x 1
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20
    best_model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, epochs, optimizer)
    plot_losses(train_losses, val_losses)
    test_model(best_model, test_loader)
    torch.save(best_model, "best_model.pt")
    
    
