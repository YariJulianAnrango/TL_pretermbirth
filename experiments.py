from training_functions import train_wo_transfer_learning, evaluate_model
import optuna
from datetime import datetime
from torch.utils.data import DataLoader

seed = 1

def objective(trial):
    params = {"batch_size": trial.suggest_int("batch_size", 2, 10),
              "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log = True),
              "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
              "layer_dim": trial.suggest_int("layer_dim", 1, 6),
              "hidden_dim": trial.suggest_int("hidden_dim", 4, 24),
              "dropout": trial.suggest_float("dropout", 0.1, 0.5),
              "epochs": trial.suggest_int("epochs", 20, 600)}
    
    train_loss, val_loss_list, val, model = train_wo_transfer_learning(params, 0.3)
    
    auc = evaluate_model(train_loss, val_loss_list, val, model)

    return auc

    
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)

best_trial = study.best_trial

now = datetime.now()

dt_string = now.strftime("%d:%m:%Y_%H:%M:%S")

file_name = "./hyperparameter_testing/parameter_testing_baseline_" + dt_string + ".txt"
with open(file_name, "w") as f:
    for key, value in best_trial.params.items():
        f.write("{}: {} ".format(key, value))
    f.write(f"auc: {study.best_value}")