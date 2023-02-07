
import torch
import torch.nn as nn
from tqdm import tqdm
import create_model
import data_setup


def training(model, epochs,criterion,optimizer,train_loader, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # Create dataset
    train_losses = []
    test_losses = []
    accuracies = []
    for epoch in tqdm(range(epochs)):
        model.train()
        for _, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%2 == 0:
            print(f"Epoch :{epoch} loss : {loss}")
            train_losses.append(loss)
        with torch.inference_mode():
            for x_test,y_test in test_loader:
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                y_predict = model(x_test)
                loss = criterion(y_predict, y_test)
            test_losses.append(loss)
    return train_losses, test_losses
