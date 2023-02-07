
import torch
import torch.nn as nn
from tqdm import tqdm

EPOCHS = 10
optimizer = torch.optim.Adam(mobilenet.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()
# Create dataset
train_loader, test_loader, classes = data_setup.create_dataset()
train_losses = []
test_losses = []
accuracies = []

def training(model,epochs=20):
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        for _, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            # predicted = torch.max(y_pred,1)[1]
            # correct = (predicted == y_train).sum()
            # accuracies.append(correct)

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
