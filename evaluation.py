import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_test, y_pred, class_names):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("📊 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_class_metrics(y_test, y_pred, class_names):
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    labels = [k for k in report.keys() if 'avg' not in k and k != 'accuracy']
    metrics = ['precision', 'recall', 'f1-score']
    data = {m: [report[l][m] for l in labels] for m in metrics}

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 5))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, data[metric], width, label=metric.capitalize())

    plt.xticks(x + width, labels, rotation=30)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("📊 Precision, Recall, F1-score per Class")
    plt.legend()
    plt.tight_layout()
    plt.show()
