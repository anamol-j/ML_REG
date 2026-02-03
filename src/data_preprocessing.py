from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def __init__(self, target_column, scale_cols, test_size=0.2):
        self.target_column = target_column
        self.scale_cols = scale_cols
        self.test_size = test_size
        self.scaler = StandardScaler()

    def split_and_scale(self,df):
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        X_train[self.scale_cols] = self.scaler.fit_transform(X_train[self.scale_cols])
        X_test[self.scale_cols] = self.scaler.transform(X_test[self.scale_cols])

        return X_train, X_test, y_train, y_test