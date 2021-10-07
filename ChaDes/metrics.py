import pandas as pd
import numpy as np

from topsis import topsis


def prepare_measures_matrix(confusion_matrix_path, classification_report_path):
    """Collect information about models.""" 
    confusion_matrix = pd.read_csv(confusion_matrix_path)
    report = pd.read_csv(classification_report_path)
    accuracy = float(report[report['Unnamed: 0']=='accuracy']['precision'])
    f1_weighted = float(report[report['Unnamed: 0']=='weighted avg']['f1-score'])
    f1_avg = float(report[report['Unnamed: 0']=='macro avg']['f1-score'])
    precision = float(report[report['Unnamed: 0']=='weighted avg']['precision'])
    recall = float(report[report['Unnamed: 0']=='weighted avg']['recall'])

    cnf_matrix = confusion_matrix.drop("Unnamed: 0", 1).to_numpy()
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    return [accuracy, f1_avg, f1_weighted, precision, recall, np.mean(TNR), np.mean(TPR), np.mean(FPR), np.mean(FNR)]


def calculate_topsis_cnn(main_path, dataset):
    """Calculate TOPSIS ranking"""
    catalogs = []
    list_measurements = []
    for color in ["rgb", "grayscale"]:
        for layer in ["2", "3", "4", "5"]:
            catalog = main_path + "color_" + color + "_frac_1.0_layers_" + layer + "_epoch_20/"
            metrics = prepare_measures_matrix(
                catalog + dataset + "_confusion_matrix.csv",
                catalog + dataset + "_classification_report.csv"
            )
            list_measurements.append(metrics)
            catalogs.append(catalog)
    decision = topsis(list_measurements, [1/9] * 9,  [1] * 7 + [0] * 2)
    b = decision.C
    return decision, b, np.argmax(decision), catalogs
