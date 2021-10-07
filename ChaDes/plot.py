import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_results(path, metric, corpora, colors, layers, number_of_labels):
    """
    Save plot with classification results.
    """
    fractions = []
    for catalog in os.listdir(path):
        splitted = catalog.split("_")
        fractions.append(splitted[3])
    fractions = sorted(np.unique(fractions))
    global rgb_l1
    rgb_l1 = [0]*len(fractions)
    global rgb_l2
    rgb_l2 = [0]*len(fractions)
    global rgb_l3
    rgb_l3 = [0]*len(fractions)
    global rgb_l4
    rgb_l4 = [0]*len(fractions)
    global rgb_l5
    rgb_l5 = [0]*len(fractions)
    global grayscale_l1
    grayscale_l1 = [0]*len(fractions)
    global grayscale_l2
    grayscale_l2 = [0]*len(fractions)
    global grayscale_l3
    grayscale_l3 = [0]*len(fractions)
    global grayscale_l4
    grayscale_l4 = [0]*len(fractions)
    global grayscale_l5
    grayscale_l5 = [0]*len(fractions)

    for idx, frac in enumerate(fractions):
        catalogs = [catalog for catalog in os.listdir(path) if "frac_"+frac in catalog]
        for color in colors:
            for layer in layers:
                catalog = [catalog for catalog in catalogs if "layers_"+layer in catalog and color in catalog]
                try:
                    report = pd.read_csv(path + "/" + catalog[0] + "/" + corpora.lower()[:4] + "_classification_report.csv")
                    x = globals()[str(color) + "_l" + str(layer)]
                    if metric == "accuracy":
                        x[idx] = report[report['Unnamed: 0']=='accuracy']['precision'][number_of_labels]
                        y_label = "Accuracy"
                        filename = "accuracy_classification_results"+corpora+".png"
                    elif metric == "f1":
                        x[idx] = report[report['Unnamed: 0']=='macro avg']["f1-score"][number_of_labels+1]
                        y_label = "F1-score"
                        filename = "f1_classification_results"+corpora+".png"
                        if layer == "5" and frac == "1.0":
                            print("PATH", path)
                            print("COLOR ", color, ", CORPORA ", corpora)
                            print("accuracy: ", report[report['Unnamed: 0']=='accuracy']['precision'][number_of_labels])
                            print("Precision, Recall, F1-score:", report[report['Unnamed: 0']=='weighted avg'])
                            print(" \n\n\n\n")
                except:
                    print(path + "/" + catalog[0] + "/" + corpora.lower()[:4] + "_classification_report.csv")
                    print("There is no report for the model: " + color + " layer_" + layer + " frac_" + str(frac))

    plt.figure(figsize=(20, 14))
    fractions = [float(frac) for frac in fractions]
    if "rgb" in colors:
        if "1" in layers:
            plt.plot(fractions, rgb_l1, marker="o", label="RGB L1", color="gray")
        if "2" in layers:
            plt.plot(fractions, rgb_l2, marker="o", label="RGB L2", color="turquoise")
        if "3" in layers:
            plt.plot(fractions, rgb_l3, marker="o", label="RGB L3", color="mediumvioletred")
        if "4" in layers:
            plt.plot(fractions, rgb_l4, marker="o", label="RGB L4", color="mediumslateblue")
        if "5" in layers:
            plt.plot(fractions, rgb_l5, marker="o", label="RGB L5", color="coral")
    if "grayscale" in colors:
        if "1" in layers:
            plt.plot(fractions, grayscale_l1,  linestyle='--', marker="o", label="GRAYSCALE L1", color="gray")
        if "2" in layers:
            plt.plot(fractions, grayscale_l2,  linestyle='--', marker="o", label="GRAYSCALE L2", color="turquoise")
        if "3" in layers:
            plt.plot(fractions, grayscale_l3,  linestyle='--', marker="o", label="GRAYSCALE L3", color="mediumvioletred")
        if "4" in layers:
            plt.plot(fractions, grayscale_l4,  linestyle='--', marker="o", label="GRAYSCALE L4", color="mediumslateblue")
        if "5" in layers:
            plt.plot(fractions, grayscale_l5,  linestyle='--', marker="o", label="GRAYSCALE L5", color="coral")

    plt.xlabel('% of training set', fontsize=26)
    plt.ylabel(y_label, fontsize=26)
    plt.title("Test Set: %s" % (corpora), fontsize=30)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=22)
    plt.savefig(
        filename,
        bbox_inches='tight',
        pad_inches=0
    )
    plt.show()
