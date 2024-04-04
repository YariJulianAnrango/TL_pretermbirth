import pandas as pd
from sklearn import metrics
import torch.nn as nn
import torch

import matplotlib.pyplot as plt

def evaluate_model(train_loss, val_loss, val, model, plot = False):
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
        sequence_shaped = sequence.unsqueeze(-1).float().to("cpu")
    
        logits_output = model(sequence_shaped)

        pred = sig(logits_output).round().int()
        print("pred",pred)
        print("label",label)
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
    
def evaluate_multiclass(train_loss, val_loss, val, model, plot = False):
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

    softmax = nn.Softmax()
    for sequence, label in val:
        sequence_shaped = sequence.unsqueeze(-1).float().to("cpu")
    
        logits_output = model(sequence_shaped)
        # print("logits \n",logits_output)
        probs = softmax(logits_output)
        # print("probs \n",probs)
        pred = torch.argmax(probs, axis = 1)
        # print("pred \n",pred)
        # print("label \n",label)
        # print()
        for p in pred:
            preds.append(p.item())
        for l in label:
            labels.append(l.item())
        
    accuracy = metrics.accuracy_score(labels, preds)
    f1_score = metrics.f1_score(labels, preds, average="weighted")

    if plot:
        print(f"accuracy: {accuracy}, f1_score {f1_score}")
        
    return f1_score

def evaluate_model_split(train_loss, val_loss, val, model, plot = False):
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
        for (x1, x2, x3), label in val:
            output = model(x1.unsqueeze(-1), x2.unsqueeze(-1), x3.unsqueeze(-1))

            pred = sig(output).round().int()
            print("pred",pred)
            print("label",label)
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
    