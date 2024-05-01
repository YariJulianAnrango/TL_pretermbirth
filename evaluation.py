import pandas as pd
from sklearn import metrics
import torch.nn as nn
import torch


import matplotlib.pyplot as plt

def evaluate_model(train_loss, val_loss, val, model, return_f1 = False, device = "cpu", plot = False):
    if plot:
        plt.plot(train_loss, label = 'train loss')
        plt.plot(val_loss, label = 'val loss')
        plt.title("Train loss and val loss per epoch")
        plt.legend()
        plt.savefig("./figures/pca_results.png")
        plt.show()

    model.eval()
    preds = []
    labels = []

    sig = nn.Sigmoid()
    for sequence, label in val:
        # sequence_shaped = sequence.unsqueeze(-1).float().to(device).permute(0,2,1)
    
        logits_output = model(sequence)

        pred = sig(logits_output).round().int()

        for p in pred:
            preds.append(p.item())
        for l in label:
            labels.append(l.item())
        
    accuracy = metrics.accuracy_score(labels, preds)
    f1_score = metrics.f1_score(labels, preds)
    
    if return_f1:
        return f1_score
    else:
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        if plot:
            plt.plot(fpr, tpr)
            plt.title(f"ROC plot with AUC {auc}, accuracy {accuracy}, and f1 score {f1_score}")
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.savefig("./figures/pca_roc.png")
            plt.show()
            print("f1_score", f1_score)
            print("accuracy", accuracy)
            print("auc", auc)
    
        return auc
    
def evaluate_multiclass(train_loss, val_loss, val, model, device = "cpu", plot = False):

    model.eval()
    preds = []
    labels = []

    softmax = nn.Softmax()
    for sequence, label in val:
        sequence_shaped = sequence.unsqueeze(-1).to(torch.float32).to(device).permute(0,2,1)
    
        logits_output = model(sequence_shaped)

        probs = softmax(logits_output)

        pred = torch.argmax(probs, axis = 1)
        for p in pred:
            preds.append(p.item())
        for l in label:
            labels.append(l.item())
        
    accuracy = metrics.accuracy_score(labels, preds)
    f1_score = metrics.f1_score(labels, preds, average="weighted")

    if plot:
        print(f"accuracy: {accuracy}, f1_score {f1_score}")
        plt.plot(train_loss, label = 'train loss')
        plt.plot(val_loss, label = 'val loss')
        plt.title("Train loss and val loss per epoch. Accuracy {accuracy}, f1 score {f1_score}")
        plt.legend()
        plt.savefig("./figures/cnn_results.png")
        plt.show()
    return f1_score

def evaluate_model_split(train_loss, val_loss, val, model, device = "cpu", plot = False):
    if plot:
        plt.plot(train_loss, label = 'train loss')
        plt.plot(val_loss, label = 'val loss')
        plt.title("Train loss and val loss per epoch")
        plt.legend()
        plt.savefig("./figures/pca_results.png")
        plt.show()

    model.eval()
    preds = []
    labels = []

    sig = nn.Sigmoid()
    with torch.no_grad():
        for x, label in val:
            output = model(x)

            pred = sig(output).round().int()

            for p in pred:
                preds.append(p.item())
            for l in label:
                labels.append(l.item())
        
    accuracy = metrics.accuracy_score(labels, preds)
    f1_score = metrics.f1_score(labels, preds)

    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    if plot:
        plt.plot(fpr, tpr)
        plt.title(f"ROC plot with AUC {auc}, accuracy {accuracy}, and f1 score {f1_score}")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.savefig("./figures/pca_roc.png")
        plt.show()
        print("f1_score", f1_score)
        print("accuracy", accuracy)
        print("auc", auc)
    
    return auc

def visualize_seq(data):
    for sequence, label in data: 
        print(sequence[0])
        print(label[0])
    