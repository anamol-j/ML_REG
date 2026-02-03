from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluation

class MLPipeline:
    def __init__(self, data_path, target_col, scale_cols):
        self.data_path = data_path
        self.target_col = target_col
        self.scale_cols = scale_cols

    def run(self):
        df = DataIngestion(self.data_path).load_data()

        X_train, X_test, y_train, y_test = DataPreprocessing(
            self.target_col, self.scale_cols
        ).split_and_scale(df)

        model, params = ModelTrainer(X_train, y_train).train()

        metrics  = ModelEvaluation().evaluate(model, X_test, y_test)

        return model, params, metrics