from model import LSTM
from model import train_and_evaluate
import optuna

seed = 1

def objective(trial):
    params = {"batch_size": trial.suggest_int("batch_size", 5, 15),
              "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log = True),
              "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
              "layer_dim": trial.suggest_int("layer_dim", 1, 6),
              "hidden_dim": trial.suggest_int("hidden_dim", 4, 24),
              "dropout": trial.suggest_float("dropout", 0.1, 0.5)}
    
    model = LSTM(params)
    
    auc = train_and_evaluate(params, model)

    return auc

    
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=30)

best_trial = study.best_trial

with open("parameter_testing.txt", "w") as f:
    for key, value in best_trial.params.items():
        f.write("{}: {} ".format(key, value))