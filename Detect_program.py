import os
import random
import time
import joblib
import numpy
import seaborn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import Func_Tool
import pandas
import xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import pyplot
from matplotlib import cm
from thundersvm import NuSVC


def evaluate_model(Y_test, Y_pred, model_name, save_folder):
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


def svm_model(X_train, X_test, Y_train, Y_test, save_folder):
    save_folder = Func_Tool.make_dir(save_folder + "SVM/")
    start_time = time.time()
    svm_model = NuSVC(kernel='rbf', gamma='auto', shrinking=True, verbose=1)
    svm_model.fit(X_train, Y_train)
    joblib.dump(svm_model, f'{save_folder}SVM_model.pkl')
    svm_Y_pred = svm_model.predict(X_test)

    evaluate_model(Y_test, svm_Y_pred, "SVM", save_folder)
    end_time = time.time()
    print(f"SVM Run time:{end_time - start_time}")


def knn_model(X_train, X_test, Y_train, Y_test, save_folder):
    save_folder = Func_Tool.make_dir(save_folder + "KNN/")
    start_time = time.time()
    knn_model = KNeighborsClassifier(n_neighbors=25, weights='uniform')

    knn_model.fit(X_train, Y_train)
    knn_y_pred = knn_model.predict(X_test)
    joblib.dump(knn_model, f'{save_folder}KNN_model.pkl')
    evaluate_model(Y_test, knn_y_pred, "KNN", save_folder)
    end_time = time.time()
    print(f"KNN Run time:{end_time - start_time}")


def rf_model(X_train, X_test, Y_train, Y_test, save_folder):
    save_folder = Func_Tool.make_dir(save_folder + "RandomForest/")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=800, random_state=0, min_samples_split=2, min_samples_leaf=1,
                                      oob_score=True)

    rf_model.fit(X_train, Y_train)
    rf_y_pred = rf_model.predict(X_test)
    joblib.dump(rf_model, f'{save_folder}RandomForest_model.pkl')
    evaluate_model(Y_test, rf_y_pred, "RandomForest", save_folder)
    end_time = time.time()
    print(f"RandomForest Run time:{end_time - start_time}")


def xgb_model(X_train, X_test, Y_train, Y_test, save_folder):
    save_folder = Func_Tool.make_dir(save_folder + "XGBoost HC3/")
    start_time = time.time()
    xbg_model = xgboost.XGBClassifier(max_depth=8, learning_rate=0.15, n_estimators=1100, verbosity=1,
                                      objective='binary:logistic', subsample=0.95, colsample_bytree=0.95,
                                      gamma=0.5, reg_alpha=0.1, reg_lambda=1,
                                      min_child_weight=1, booster='gbtree', eval_metric='logloss')
    xbg_model.fit(X_train, Y_train)
    joblib.dump(xbg_model, f'{save_folder}XGBoost_model.pkl')
    xgb_Y_pred = xbg_model.predict(X_test)
    evaluate_model(Y_test, xgb_Y_pred, "XGBoost", save_folder)
    end_time = time.time()
    print("XGBoost Run time:", end_time - start_time)


def read_result(M, L, N, param_score_map):
    values = numpy.empty(M.shape)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                key = (M[i, j, k], L[i, j, k], N[i, j, k])
                values[i, j, k] = param_score_map.get(key, numpy.nan)
    return values


def XGB_GridSearch(X_train, X_test, Y_train, Y_test, save_folder):
    save_folder = Func_Tool.make_dir(save_folder + "GridSearch XGB model/")
    params_grid = {'max_depth': [24, 25, 26, 27, 28, 29, 30],
                   'learning_rate': [0.05, 0.03, 0.01, 0.005, 0.001],
                   'n_estimators': [1600, 1700, 1800, 1900, 2000, 2100],
                   }

    xbg_model = xgboost.XGBClassifier(verbosity=1,
                                      objective='binary:logistic', scale_pos_weight=1,
                                      subsample=0.95, colsample_bytree=0.95, gamma=0.5, reg_alpha=0.1, reg_lambda=1,
                                      min_child_weight=1, booster='gbtree', eval_metric='logloss')
    grid_search = GridSearchCV(estimator=xbg_model, param_grid=params_grid, cv=3, n_jobs=4)
    grid_search.fit(X_train, Y_train)
    results = grid_search.cv_results_
    Func_Tool.write_txt_data(str(results), "XGBmodel_grid_search_results", save_folder)

    max_depth = params_grid['max_depth']
    learning_rate = params_grid['learning_rate']
    n_estimators = params_grid['n_estimators']
    params = results['params']
    mean_test_score = results['mean_test_score']
    rank_test_score = results['rank_test_score']

    Func_Tool.write_txt_data(str(list(params)), "XGBmodel_grid_search_params", save_folder)
    Func_Tool.write_txt_data(str(list(mean_test_score)), "XGBmodel_grid_search_mean_test_score", save_folder)
    Func_Tool.write_txt_data(str(list(rank_test_score)), "XGBmodel_grid_search_rank_test_score", save_folder)


    param_score_map = {
        (p['max_depth'], p['learning_rate'], p['n_estimators']): score
        for p, score in zip(params, mean_test_score)
    }
    M, L, N = numpy.meshgrid(max_depth, learning_rate, n_estimators, indexing='ij')
    values = read_result(M, L, N, param_score_map)
    n_samples = values.size
    indices = numpy.random.choice(numpy.arange(values.size), n_samples, replace=False)

    M_sample = M.flatten()[indices]
    L_sample = L.flatten()[indices]
    N_sample = N.flatten()[indices]
    V_sample = values.flatten()[indices]

    max_idx = numpy.argmax(V_sample)
    M_max, L_max, N_max = M_sample[max_idx], L_sample[max_idx], N_sample[max_idx]
    V_max = V_sample[max_idx]

    figure = pyplot.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection='3d')

    norm = pyplot.Normalize(V_sample.min(), V_sample.max())
    colors = cm.viridis(norm(V_sample))

    scatter = axis.scatter(M_sample, L_sample, N_sample, c=V_sample, cmap='viridis')

    axis.scatter(M_max, L_max, N_max, color='red', s=100,
                 label=f'Max Value ({V_max:.2f})\nCoordinates: ({M_max:.2f}, {L_max:.2f}, {N_max:.2f})')

    cbar = figure.colorbar(scatter, ax=axis, shrink=0.6, aspect=10)
    cbar.set_label('mean_test_score')

    axis.set_xlabel('max_depth')
    axis.set_ylabel('learning_rate')
    axis.set_zlabel('n_estimators')
    axis.set_title('3D Scatter Plot with Color Mapping')

    axis.legend()
    pyplot.savefig(f"{save_folder}XGBmodel_grid_search.png")
    pyplot.show()

    best_model = grid_search.best_estimator_

    test_score = best_model.score(X_test, Y_test)

    print("Best Parameters: ", grid_search.best_params_)
    information = f"Best Parameters: {grid_search.best_params_}\nBest Score: {grid_search.best_score_}\nTest Score: {test_score}"
    Func_Tool.write_txt_data(information, f"xgb_GridSearch_information", save_folder)

    joblib.dump(best_model, save_folder + "best-xgb-model.pkl")

    best_model = joblib.load(save_folder + "best-xgb-model.pkl")
    xgb_Y_pred = best_model.predict(X_test)

    evaluate_model(Y_test, xgb_Y_pred, "best-xgb-model", save_folder)



if __name__ == '__main__':
    save_folder = Func_Tool.make_dir(f"./result/classify/")
    Cn_dataset = pandas.read_csv(f"./result/features/Cn_features.csv", header=0, dtype=float).values
    En_dataset = pandas.read_csv(f"./result/features/En_features.csv", header=0, dtype=float).values
    faketext_dataset = pandas.read_csv(f"./result/features/faketext_features.csv", header=0, dtype=float).values
    dataset = numpy.concatenate((Cn_dataset, En_dataset, faketext_dataset), axis=0)
    character_num = 11


    random.seed(0)
    random.shuffle(dataset)

    train_Data = dataset[:, 0:character_num]
    train_Label = dataset[:, -1]

    print(f"HWT number:{sum(train_Label == 1)}")
    print(f"MGT number:{sum(train_Label == 0)}")

    std_scaler = StandardScaler()
    std_scaler.fit(train_Data)

    X_train, X_test, Y_train, Y_test = train_test_split(train_Data, train_Label, random_state=1, shuffle=True,
                                                        test_size=0.2)
    print(f"train number: {len(X_train)}")
    print(f"test number: {len(X_test)}")

    start_time = time.time()

    svm_model(X_train, X_test, Y_train, Y_test, save_folder)
    knn_model(X_train, X_test, Y_train, Y_test, save_folder)
    rf_model(X_train, X_test, Y_train, Y_test, save_folder)
    xgb_model(X_train, X_test, Y_train, Y_test, save_folder)
    XGB_GridSearch(X_train, X_test, Y_train, Y_test, save_folder)

    end_time = time.time()
    print(f"Run time:{end_time - start_time}")

