import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def objective(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "random_state": 42
        }

        model = GradientBoostingRegressor(**params)

        score = cross_val_score(
            model, self.X_train, self.y_train,
            cv=5, scoring="r2"
        ).mean()

        return score
    
    def train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=50)

        best_model = GradientBoostingRegressor(
            **study.best_params,
            random_state=42
        )
        best_model.fit(self.X_train, self.y_train)

        return best_model, study.best_params