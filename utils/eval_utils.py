from sklearn.metrics import average_precision_score, accuracy_score, f1_score, classification_report
import torch
def acc_f1(output, labels, average='macro'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(preds, labels, average=average, zero_division=0)
    report = classification_report(preds, labels, digits=3, zero_division=0)
    return accuracy, f1, report