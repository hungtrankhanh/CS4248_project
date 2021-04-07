from sklearn.model_selection import train_test_split
from torch import optim
import torch
from .MLP import IntervenionModel
from .preprocessing import *
import torch.nn.functional as F

def plot_line(title, xlabel1, ylabel1, xlabel2, ylabel2, indices, y11, y12, y21, y22):
    pass
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_figheight(5)
    # fig.set_figwidth(10)
    # fig.suptitle(title)
    # fig.subplots_adjust(wspace=0.25)
    # ax1.plot(indices, y11)
    # ax1.plot(indices, y12)
    # ax1.set(xlabel=xlabel1, ylabel=ylabel1)

    # ax2.plot(indices, y21)
    # ax2.plot(indices, y22)
    # ax2.set(xlabel=xlabel2, ylabel=ylabel2)

    # ax1.legend(['Reinforce', 'Random'])
    # ax2.legend(['Reinforce', 'Random'])
    # # plt.legend()
    # # plt.show()
    # plt.savefig('acc_loss.png')

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
        print("train_count : ", train_count)
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

        print(f'\tTrain Loss: {epoch_train_loss:.3f} | Train Acc: {epoch_train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {epoch_val_loss:.3f} |  Val. Acc: {epoch_val_acc * 100:.2f}%')

    return model

def predict(model, X):
    X_test = torch.tensor(X, dtype=torch.float32).to(device)
    prediction = model(X_test)
    prediction = prediction.detach().cpu().numpy()
    return prediction.flatten()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Instantiate the model with hyperparameters
    model = IntervenionModel(33, 16)
    # We'll also set the model to the device that we defined earlier (default is CPU)
    model = model.to(device)
    X_train_data, y_train_label = loadDatasets("train")
    X_test_data, y_test_label = loadDatasets("test")

    # Define hyperparameters
    n_epochs = 100
    lr = 0.001

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    print("-----------train")
    model = train(model, X_train_data,y_train_label, optimizer, criterion, device)

    preds = predict(model, X_test_data)
    print("-----------predict : ", preds)

    # Use f1-macro as the metric
    score = f1_score(y_test_label, preds.round(), average='macro')
    print('score on validation = {}'.format(score))