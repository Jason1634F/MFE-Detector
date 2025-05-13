import os
import random
import pandas
import seaborn
import torch
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import Func_Tool


class BinaryClassificationModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512, bias=True)
        self.fc2 = torch.nn.Linear(512, 128, bias=True)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc3 = torch.nn.Linear(128, 16, bias=True)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.fc4 = torch.nn.Linear(16, 1, bias=True)
        self.dropout = torch.nn.Dropout(0.1)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim = 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.bn1(x)
        x = self.relu(self.fc3(x))
        x = self.bn2(x)
        x = self.relu(self.fc4(x))
        x = self.sigmoid(x)
        return x

def load_dataloader():
    Cn_dataset = pandas.read_csv(f"./result/features/Cn_features.csv", header=0, dtype=float).values
    En_dataset = pandas.read_csv(f"./result/features/En_features.csv", header=0, dtype=float).values
    faketext_dataset = pandas.read_csv(f"./result/features/faketext_features.csv", header=0, dtype=float).values

    dataset = numpy.concatenate((Cn_dataset, En_dataset, faketext_dataset), axis=0)
    character_num = 11

    X_narray = dataset[:,0:character_num]
    y_narray = dataset[:,-1]

    scaler = StandardScaler()
    X_narray = scaler.fit_transform(X_narray)

    X_tensor = torch.tensor(X_narray, dtype=torch.float32)
    y_tensor = torch.tensor(y_narray, dtype=torch.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train, X_test, y_test, device):
    save_dir = Func_Tool.make_dir("./result/classify/mlp classify/")
    save_path = os.path.join(save_dir, 'mlp_model.pth')

    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    num_epochs = 100
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        epoch_loss = loss.item()

        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            predicted = (outputs > 0.5).float()
            accuracy = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)

        print(f'Epoch {epoch + 1}/{num_epochs}: Loss: {epoch_loss:.4f}; Accuracy: {accuracy:.4f}')
        scheduler.step()


def test_model(model, X_test, y_test, device):
    model_dir = './result/classify/mlp classify/mlp_model.pth'
    model.load_state_dict(torch.load(model_dir))

    model.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs > 0.5).float()
        accuracy = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
    print(f'Accuracy on test dataset: {accuracy:.4f}')

    save_folder = r'./result/classify/mlp classify/'
    Y_test = y_test.cpu().numpy()
    Y_pred = predicted.cpu().numpy()

    matrix = confusion_matrix(Y_test, Y_pred, labels=[0, 1])
    True_Positives = matrix[1, 1]
    True_Negatives = matrix[0, 0]
    False_Positives = matrix[0, 1]
    False_Negatives = matrix[1, 0]

    accuracy = accuracy_score(Y_test, Y_pred)
    recall = True_Positives / (True_Positives + False_Negatives)
    precision = True_Positives / (True_Positives + False_Positives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    roc_auc = roc_auc_score(Y_test, Y_pred)

    model_name = "MLP"
    information = f"True_Positives: {True_Positives}\nTrue_Negatives: {True_Negatives}\n" \
                  f"False_Positives: {False_Positives}\nFalse_Negatives: {False_Negatives}\n" \
                  f"accuracy: {accuracy}\nrecall: {recall}\nprecision: {precision}\n" \
                  f"F1-score: {f1_score}\nroc_auc : {roc_auc}\n"
    Func_Tool.write_txt_data(information, f"{model_name}_information", save_folder)

    print(f"{model_name}accuracy: {accuracy}")

    heatmap_matrix = pandas.DataFrame(data=matrix,
                                      columns=['Predict Negative:0', 'Predict Positive :1'],
                                      index=['Actual Negative:0', 'Actual Positive:1'])
    seaborn.heatmap(heatmap_matrix, annot=True, fmt='d', cmap='YlGnBu')
    pyplot.xlabel("predict label")
    pyplot.ylabel('real label')
    pyplot.title(f"{model_name} accuracy: {accuracy}")
    pyplot.savefig(f"{save_folder}{model_name}_heatmap.png")
    pyplot.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test, y_train, y_test = load_dataloader()


    feature_dim = 11
    model = BinaryClassificationModel(input_dim=feature_dim)
    train_model(model, X_train, y_train, X_test, y_test, device)
    test_model(model, X_test, y_test, device)
