import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, fbeta_score
import torch
from torch import optim
import random
import sys
import numpy as np

from preprocessing import *
from mlp import *

torch.manual_seed(4248)
np.random.seed(4248)
random.seed(4248)
np.set_printoptions(threshold=sys.maxsize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='binary')

def train(model,X,Y, optimizer, criterion, device, batch_size=32, n_epochs = 20):
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=4248)
    print("X_train shape = ", X_train.shape)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    data_train = torch.utils.data.TensorDataset(X_train, y_train)
    data_val = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False)

    for epoch in range(n_epochs):
        epoch_train_loss = 0
        epoch_train_acc = 0
        # set the model in training phase
        model.train()
        train_count = 0
        for i, (inputs, labels) in enumerate(train_loader):
            train_count += len(inputs)
            # resets the gradients after every batch
            optimizer.zero_grad()
            # convert to 1D tensor
            predictions = model(inputs)
            labels = labels.reshape(-1,1)
            # compute the loss
            loss = criterion(predictions, labels)
            # backpropage the loss and compute the gradients
            loss.backward()

            # update the weights
            optimizer.step()
            rounded_preds = torch.round(predictions)
            correct = (rounded_preds == labels).float()
            acc = correct.sum() / len(correct)
            # loss and accuracy
            epoch_train_loss += loss.item()
            epoch_train_acc += acc.item()
        # print("train_count : ", train_count)
        epoch_train_loss = epoch_train_loss / len(train_loader)
        epoch_train_acc = epoch_train_acc / len(train_loader)

        # initialize every epoch
        epoch_val_loss = 0
        epoch_val_acc = 0

        # deactivating dropout layers
        model.eval()

        # deactivates autograd
        with torch.no_grad():
            for i ,(val_inputs, val_labels) in enumerate(valid_loader):
                # convert to 1D tensor
                predictions = model(val_inputs)
                val_labels = val_labels.reshape(-1,1)

                # compute the loss
                loss = criterion(predictions, val_labels)

                rounded_preds = torch.round(predictions)
                correct = (rounded_preds == val_labels).float()
                acc = correct.sum() / len(correct)
                # loss and accuracy
                epoch_val_loss += loss.item()
                epoch_val_acc += acc.item()
            epoch_val_loss = epoch_val_loss / len(valid_loader)
            epoch_val_acc = epoch_val_acc / len(valid_loader)

        print(f'\tepoch: {epoch}/{n_epochs} | Train Loss: {epoch_train_loss:.3f} | Train Acc: {epoch_train_acc * 100:.2f}%')
        print(f'\tepoch: {epoch}/{n_epochs} | Val. Loss: {epoch_val_loss:.3f} |  Val. Acc: {epoch_val_acc * 100:.2f}%')

    return model

def predict(model, X):
    X_test = torch.tensor(X, dtype=torch.float32).to(device)
    prediction = model(X_test)
    prediction = prediction.detach().cpu().numpy()
    return prediction.flatten()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='The instructor intervention mode')
    args.add_argument('--feature','-feature', help="tf_idf or word2vec model")

    args.add_argument('--dataset','-dataset', help="raw or processed")
    args = args.parse_args()
    print("arg :", args)

    X_train_data, y_train_label, X_test_data, y_test_label = load_datasets(args)
    d_in = X_train_data.shape[1]
    # Instantiate the model with hyperparameters
    model = IntervenionModel(d_in, 12)
    # We'll also set the model to the device that we defined earlier (default is CPU)
    model = model.to(device)

    # Define hyperparameters
    n_epochs = 50
    lr = 0.0001

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model = train(model, X_train_data,y_train_label, optimizer, criterion, device,batch_size=16, n_epochs=n_epochs)

    preds = predict(model, X_test_data)
    print("-----------prediction evaluation metrics:")

    # Use f1-macro as the metric
    score = f1_score(y_test_label, preds.round(), average='binary')
    print('f1 score on validation = {}'.format(score))
    print('accuracy score on validation = {}'.format(accuracy_score(y_test_label, preds.round())))
    print('precision score on validation = {}'.format(precision_score(y_test_label, preds.round(), average='binary')))
    print('recall score on validation = {}'.format(recall_score(y_test_label, preds.round(), average='binary')))
    print('f2 score on validation = {}'.format(f2_score(y_test_label, preds.round())))


