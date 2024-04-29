from training_functions import (
    pretrain,
    get_ucr_data,
    perform_transfer_learning,
)
from evaluation import evaluate_model, evaluate_multiclass, evaluate_model_split
from multichannel_concatenation import multichannel_finetune, kfold_multichannel, load_pretrained_lstm
from model import LSTM, MultiChannelModel, CNN

import optuna
from datetime import datetime
from functools import partial

import os

seed = 1


def objective(trial):
    params = {
        "batch_size": trial.suggest_int("batch_size", 2, 20),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "layer_dim": trial.suggest_int("layer_dim", 1, 6),
        "hidden_dim": trial.suggest_int("hidden_dim", 4, 24),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "epochs": trial.suggest_int("epochs", 20, 600),
    }
    train, val, test, target_size = get_ucr_data(
        "/Users/yarianrango/Documents/School/Master-AI-VU/Thesis/data/UCRArchive_2018/ECG5000/ECG5000_TRAIN.tsv",
        0.3,
    )

    train_loss, val_loss_list, val, model = pretrain(train, val, target_size, params)

    auc = evaluate_multiclass(train_loss, val_loss_list, val, model)

    return auc

def tl_objective(trial):
    params = {
        "batch_size": trial.suggest_int("batch_size", 2, 20),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "layer_dim": trial.suggest_int("layer_dim", 1, 6),
        "hidden_dim": trial.suggest_int("hidden_dim", 4, 24),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "epochs": trial.suggest_int("epochs", 20, 600),
        "batch_size_fine": trial.suggest_int("batch_size", 2, 20),
        "learning_rate_fine": trial.suggest_float(
            "learning_rate_fine", 1e-5, 1e-1, log=True
        ),
        "optimizer_fine": trial.suggest_categorical(
            "optimizer_fine", ["Adam", "RMSprop", "SGD"]
        ),
        "epochs_fine": trial.suggest_int("epochs_fine", 20, 600),
    }

    fine_train_loss, fine_val_loss, preterm_val, finetuned_model = (
        perform_transfer_learning(
            "/Users/yarianrango/Documents/School/Master-AI-VU/Thesis/data/UCRArchive_2018/ECG5000/ECG5000_TRAIN.tsv",
            0.3,
            params,
        )
    )

    auc = evaluate_model(fine_train_loss, fine_val_loss, preterm_val, finetuned_model)

    return auc

def multichannel_tf_objective(trial):
    params = {
        "batch_size": trial.suggest_int("batch_size", 2, 20),
        "learning_rate": trial.suggest_float( "learning_rate", 1e-5, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "epochs": trial.suggest_int("epochs", 20, 30),
        "lin_layers": trial.suggest_int("lin_layers", 1, 3),
        "hidden1": trial.suggest_int("hidden1", 4, 256),
        "hidden2": trial.suggest_int("hidden2", 4, 256),
        "loss_weight": trial.suggest_float("loss_weight", 1, 10)
    }

    train_loss, val_loss_list, val, mLSTM = multichannel_finetune(params_fine = params, device = "cpu")
    
    auc = evaluate_model_split(train_loss, val_loss_list, val, mLSTM, device = "cpu")

    return auc

def pretrain_objective(trial, data_path):
    params = {
        "batch_size": trial.suggest_int("batch_size", 4, 64),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "layer_dim": trial.suggest_int("layer_dim", 1, 4),
        "hidden_dim": trial.suggest_int("hidden_dim", 4, 24),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "epochs": trial.suggest_int("epochs", 20, 200),
    }
    
    device = "cpu"
    source_train, source_val, target_size = get_ucr_data(data_path, 0.3, params)
    cnn = CNN(target_size, device)
    if target_size == 2:
        target_size = 1  
        train_loss, val_loss, val, model = pretrain(cnn, source_train, source_val, target_size, params, device, binary_classes=True)
        f1_score = evaluate_model(train_loss, val_loss, val, model, return_f1=True, device = device)
    else:
        train_loss, val_loss, val, model = pretrain(cnn, source_train, source_val, target_size, params, device)
        f1_score = evaluate_multiclass(train_loss, val_loss, val, model, device)
    
    return f1_score

def pretrain_cnn_objective(trial, data_path):
    params = {
        "batch_size": trial.suggest_int("batch_size", 4, 64),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "epochs": trial.suggest_int("epochs", 20, 200),
    }
    
    device = "cpu"
    source_train, source_val, target_size = get_ucr_data(data_path, 0.3, params)
    if target_size == 2:
        target_size = 1  
        cnn = CNN(target_size)
        train_loss, val_loss, val, model = pretrain(cnn, source_train, source_val, target_size, params, device, binary_classes=True)
        f1_score = evaluate_model(train_loss, val_loss, val, model, return_f1=True, device = device)
    else:
        cnn = CNN(target_size)
        train_loss, val_loss, val, model = pretrain(cnn, source_train, source_val, target_size, params, device)
        f1_score = evaluate_multiclass(train_loss, val_loss, val, model, device)
    
    return f1_score

def kfold_mLSTM_objective(trial):
    params = {
        "batch_size": trial.suggest_int("batch_size", 2, 20),
        "learning_rate": trial.suggest_float( "learning_rate", 1e-5, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "epochs": trial.suggest_int("epochs", 20, 100),
        "lin_layers": trial.suggest_int("lin_layers", 1, 3),
        "hidden1": trial.suggest_int("hidden1", 4, 256),
        "hidden2": trial.suggest_int("hidden2", 4, 256),
        "loss_weight": trial.suggest_float("loss_weight", 1, 10)
    }

    auc = kfold_multichannel(params_fine = params, device = "cpu")

    return auc

# def kfold_objective(trial, model_name, model_path, target_size, params_pre):
#    params = {
#         "batch_size": trial.suggest_int("batch_size", 2, 20),
#         "learning_rate": trial.suggest_float( "learning_rate", 1e-5, 1e-1, log=True),
#         "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
#         "epochs": trial.suggest_int("epochs", 20, 100),
#         "lin_layers": trial.suggest_int("lin_layers", 1, 3),
#         "hidden1": trial.suggest_int("hidden1", 4, 256),
#         "hidden2": trial.suggest_int("hidden2", 4, 256),
#         "loss_weight": trial.suggest_float("loss_weight", 1, 10)
#     }

#     auc = kfold_multichannel(model_name, model_path, target_size, params_fine = params, device = "cpu", params_pre=params_pre)

#     return auc 

def tune_all_source_data(source_data_directory):
    for folder_name in os.listdir(source_data_directory):
        folder_path = os.path.join(source_data_directory, folder_name)
        if os.path.isdir(folder_path):  
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.tsv') and 'TRAIN' in file_name:
                    print(f"Filename: {file_name}:")
                    file_path = os.path.join(folder_path, file_name)
                    
                    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
                    objective = partial(pretrain_cnn_objective, data_path = file_path)
                    study.optimize(objective, n_trials=25)

                    best_trial = study.best_trial

                    now = datetime.now()

                    dt_string = now.strftime("%d:%m:%Y_%H:%M:%S")

                    file_name = "./hyperparameter_testing/cnn_source_data/parameter_testing_" + file_name + "_" + dt_string + ".txt"
                    with open(file_name, "w") as f:
                        for key, value in best_trial.params.items():
                            f.write("{}: {} ".format(key, value))
                        f.write(f"f1: {study.best_value}")
                        
# def kfold_objective(trial, model_name, model_path, target_size, params_pre):
#    params = {
#         "batch_size": trial.suggest_int("batch_size", 2, 20),
#         "learning_rate": trial.suggest_float( "learning_rate", 1e-5, 1e-1, log=True),
#         "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
#         "epochs": trial.suggest_int("epochs", 20, 100),
#         "lin_layers": trial.suggest_int("lin_layers", 1, 3),
#         "hidden1": trial.suggest_int("hidden1", 4, 256),
#         "hidden2": trial.suggest_int("hidden2", 4, 256),
#         "loss_weight": trial.suggest_float("loss_weight", 1, 10)
#     }
#     auc = kfold_multichannel(model_name, model_path, target_size, params_fine = params, device = "cpu", params_pre=params_pre)
#     return auc 

# def tune_all_target_data(model_name, pretrained_dir, source_data_dir):
#     for trained_model in os.listdir(pretrained_dir):
#         for source_data in os.listdir(source_data_dir):
#             if os.path.isdir(source_data):  
#                 folder_path = os.path.join(source_data_dir, source_data)
#                 for file_name in os.listdir(source_data):
#                     if file_name in trained_model:
#                         source_dir = os.path.join(folder_path, file_name)
#                         source_train, source_val, target_size = get_ucr_data(source_dir, 0.3, params_pre)
#                         model_path = os.path.join(pretrained_dir, trained_model)
        
#                         study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
#                         objective = partial(kfold_objective, model_name = model_name, model_path = model_path)
#                         study.optimize(objective, n_trials=25)

#                         best_trial = study.best_trial

#                         now = datetime.now()

#                         dt_string = now.strftime("%d:%m:%Y_%H:%M:%S")

#                         file_name = "./hyperparameter_testing/cnn_source_data/parameter_testing_" + file_name + "_" + dt_string + ".txt"
#                         with open(file_name, "w") as f:
#                             for key, value in best_trial.params.items():
#                                 f.write("{}: {} ".format(key, value))
#                             f.write(f"f1: {study.best_value}")
#         if os.path.isdir(folder_path):  
#             for file_name in os.listdir(folder_path):
#                 if file_name.endswith('.tsv') and 'TRAIN' in file_name:
#                     print(f"Filename: {file_name}:")
#                     file_path = os.path.join(folder_path, file_name)
                    
#                     study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
#                     objective = partial(pretrain_cnn_objective, data_path = file_path)
#                     study.optimize(objective, n_trials=25)

#                     best_trial = study.best_trial

#                     now = datetime.now()

#                     dt_string = now.strftime("%d:%m:%Y_%H:%M:%S")

#                     file_name = "./hyperparameter_testing/cnn_source_data/parameter_testing_" + file_name + "_" + dt_string + ".txt"
#                     with open(file_name, "w") as f:
#                         for key, value in best_trial.params.items():
#                             f.write("{}: {} ".format(key, value))
#                         f.write(f"f1: {study.best_value}")

def finetune_cnn(trial):
    params = {
        "batch_size": trial.suggest_int("batch_size", 2, 20),
        "learning_rate": trial.suggest_float( "learning_rate", 1e-5, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "epochs": trial.suggest_int("epochs", 20, 100),
        "lin_layers": trial.suggest_int("lin_layers", 1, 3),
        "hidden1": trial.suggest_int("hidden1", 4, 256),
        "hidden2": trial.suggest_int("hidden2", 4, 256)
    }
    auc = kfold_multichannel("CNN", "./models/pretrained_cnns/ECG5000_TRAIN.tsv.pt", 42, params_fine = params, device = "cpu", params_pre=None)
    
    return auc 

# tune_all_source_data("../data/source_datasets")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(finetune_cnn, n_trials=25)

best_trial = study.best_trial

now = datetime.now()

dt_string = now.strftime("%d:%m:%Y_%H:%M:%S")

file_name = "./hyperparameter_testing/cnn_kfold/parameter_testing_CNN_kfold_ECG5000_" + dt_string + ".txt"
with open(file_name, "w") as f:
    for key, value in best_trial.params.items():
        f.write("{}: {} ".format(key, value))
    f.write(f"auc: {study.best_value}")
    

