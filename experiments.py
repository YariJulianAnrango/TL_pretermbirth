from training_functions import (
    evaluate_model,
    pretrain,
    evaluate_multiclass,
    get_ucr_data,
    perform_transfer_learning,
)
import optuna
from datetime import datetime

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


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(tl_objective, n_trials=50)

best_trial = study.best_trial

now = datetime.now()

dt_string = now.strftime("%d:%m:%Y_%H:%M:%S")

file_name = "./hyperparameter_testing/parameter_testing_tl_" + dt_string + ".txt"
with open(file_name, "w") as f:
    for key, value in best_trial.params.items():
        f.write("{}: {} ".format(key, value))
    f.write(f"auc: {study.best_value}")
