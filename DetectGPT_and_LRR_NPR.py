import pandas
import seaborn
import tqdm
from matplotlib import pyplot
import Func_Tool
import numpy
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def evaluate_model(Y_test, Y_pred, threshold, model_name, save_folder):
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
                  f"Boundary threshold:{threshold}\naccuracy: {accuracy}\nrecall: {recall}\nprecision: {precision}\n" \
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


def get_best_threshold(features, labels, method_name):
    save_folder = Func_Tool.make_dir("./result/baseline_experiment/"+method_name+'/')
    best_threshold = None
    best_accuracy = 0


    step_size = 0.00001
    min_threshold = numpy.min(features)
    print(f"{method_name} min_threshold: ", min_threshold)
    max_threshold = numpy.max(features)
    print(f"{method_name} max_threshold: ", max_threshold)

    thresholds = []
    accuracies = []

    Early_Stopping = 0.05

    for threshold in tqdm.tqdm(numpy.arange(min_threshold, max_threshold, step_size)):
        predicted_labels = (features > threshold).astype(int)
        accuracy = accuracy_score(labels, predicted_labels)
        thresholds.append(threshold)
        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

        if best_accuracy - accuracy > Early_Stopping:
            break

    pyplot.plot(thresholds, accuracies, label="Accuracy vs Threshold", color='green')
    pyplot.xlabel("Threshold")
    pyplot.ylabel("Accuracy")
    pyplot.title("Accuracy as Threshold")
    pyplot.savefig(f"{save_folder}{method_name} Accuracy as Threshold.png")
    pyplot.legend()
    pyplot.show()

    print(f"{method_name} Best Threshold:  \t{best_threshold}")
    print(f"{method_name} The highest accuracy rate: \t{best_accuracy}")

    predicted_labels = (features > best_threshold).astype(int)
    evaluate_model(labels, predicted_labels, best_threshold, method_name, save_folder)


if __name__ == '__main__':
    DetectGPT_data = pandas.read_csv(f"./result/perturb_features/DetectGPT_data.csv", dtype=float).values
    DetectGPT_feature = DetectGPT_data[:, 0]
    DetectGPT_label = DetectGPT_data[:, 1]
    get_best_threshold(DetectGPT_feature, DetectGPT_label, "DetectGPT")

    DetectLLM_LRR_data = pandas.read_csv(f"./result/perturb_features/DetectLLM_LRR_data.csv", dtype=float).values
    DetectLLM_LRR_feature = DetectLLM_LRR_data[:, 0]
    DetectLLM_LRR_label = DetectLLM_LRR_data[:, 1]
    get_best_threshold(DetectLLM_LRR_feature, DetectLLM_LRR_label, "DetectLLM_LRR")

    DetectLLM_NPR_data = pandas.read_csv(f"./result/perturb_features/DetectLLM_NPR_data.csv", dtype=float).values
    DetectLLM_NPR_feature = DetectLLM_NPR_data[:, 0]
    DetectLLM_NPR_label = DetectLLM_NPR_data[:, 1]
    get_best_threshold(DetectLLM_NPR_feature, DetectLLM_NPR_label, "DetectLLM_NPR")


