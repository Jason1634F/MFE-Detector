import random
import pandas
import seaborn
import torch
import xgboost
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import Func_Tool
import os
import numpy
from transformers import logging
from sklearn.linear_model import LogisticRegression
import sys
import thundersvm
from thundersvm import NuSVC

logging.set_verbosity_error()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def evaluate_model(Y_test, Y_pred, index_name, save_folder):
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

    information = f"True_Positives: {True_Positives}\nTrue_Negatives: {True_Negatives}\n" \
                  f"False_Positives: {False_Positives}\nFalse_Negatives: {False_Negatives}\n" \
                  f"accuracy: {accuracy}\nrecall: {recall}\nprecision: {precision}\n" \
                  f"F1-score: {f1_score}\nroc_auc : {roc_auc}\n"
    Func_Tool.write_txt_data(information, f"{index_name}_information", save_folder)
    print(f"{index_name}accuracy: {accuracy}")
    heatmap_matrix = pandas.DataFrame(data=matrix,
                                      columns=['Predict Negative:0', 'Predict Positive :1'],
                                      index=['Actual Negative:0', 'Actual Positive:1'])
    seaborn.heatmap(heatmap_matrix, annot=True, fmt='d', cmap='YlGnBu')
    pyplot.xlabel("predict label")
    pyplot.ylabel('real label')
    pyplot.title(f"{index_name} accuracy: {accuracy}")
    pyplot.savefig(f"{save_folder}{index_name}_heatmap.png")
    pyplot.show()


def run_baseline_experiment(name, data, label, save_folder):
    save_folder = Func_Tool.make_dir(save_folder + f"{name}/")
    X_train, X_test, Y_train, Y_test = train_test_split(data, label, random_state=1, test_size=0.1)
    model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', class_weight='balanced', warm_start=False)
    model.fit(X_train, Y_train)
    joblib.dump(model, f'{save_folder}{name}_model.pkl')
    Y_pred = model.predict(X_test)
    evaluate_model(Y_test, Y_pred, name, save_folder)


def run_supervised_experiment(name, data, label, save_folder):
    save_folder = Func_Tool.make_dir(save_folder + f"{name}/")
    X_train, X_test, Y_train, Y_test = train_test_split(data, label, random_state=1, test_size=0.1)
    train_result = []
    test_result = []
    for i in X_train:
        if i < 0.5:
            train_result.append(0)
        else:
            train_result.append(1)
    for i in X_test:
        if i < 0.5:
            test_result.append(0)
        else:
            test_result.append(1)
    evaluate_model(Y_test, test_result, name, save_folder)


if __name__ == '__main__':
    save_folder = Func_Tool.make_dir(f"./result/baseline_experiment/")

    Cn_dataset = pandas.read_csv(f"./result/features/Cn_features.csv", header=0, dtype=float).values
    En_dataset = pandas.read_csv(f"./result/features/En_features.csv", header=0, dtype=float).values
    faketext_dataset = pandas.read_csv(f"./result/features/faketext_features.csv", header=0, dtype=float).values

    dataset = numpy.concatenate((Cn_dataset, En_dataset, faketext_dataset), axis=0)

    random.seed(0)
    random.shuffle(dataset)

    character_num = 11
    train_Label = dataset[:, -1]
    print(f"HWT number:{sum(train_Label == 1)}")
    print(f"MGT number:{sum(train_Label == 0)}")

    entropy = dataset[:,0].reshape(-1, 1)
    rank = dataset[:,1].reshape(-1, 1)
    log_rank = dataset[:,2].reshape(-1, 1)
    likelihood = dataset[:,5].reshape(-1, 1)
    supervised_result = dataset[:,10].reshape(-1, 1)
    train_Label = dataset[:,-1]


    std_scaler = StandardScaler()
    std_scaler.fit(entropy)
    std_scaler.fit(rank)
    std_scaler.fit(log_rank)
    std_scaler.fit(likelihood)


    run_baseline_experiment("entropy", entropy, train_Label, save_folder)
    run_baseline_experiment("rank", rank, train_Label, save_folder)
    run_baseline_experiment("log_rank", log_rank, train_Label, save_folder)
    run_baseline_experiment("likelihood", likelihood, train_Label, save_folder)
    run_supervised_experiment("supervised_result", supervised_result, train_Label, save_folder)