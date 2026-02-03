from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class ModelEvaluation:
    def evaluate(self, model, X_test, y_test):
        preds = model.predict(X_test)

        return {
            "R2": r2_score(y_test, preds),
            "RMSE": mean_squared_error(y_test, preds),
            "MAE": mean_absolute_error(y_test, preds)
        }