import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessing:
    def __init__(
        self,
        target_column,
        scale_cols=None,
        cat_cols=None,
        test_size=0.2,
        num_impute="mean",
        cat_impute="mode",
        cat_constant="Unknown"
    ):
        self.target_column = target_column
        self.scale_cols = scale_cols or []
        self.cat_cols = cat_cols
        self.test_size = test_size
        self.num_impute = num_impute
        self.cat_impute = cat_impute
        self.cat_constant = cat_constant
        self.scaler = StandardScaler()

    def split_and_scale(self,df):
        df = df.copy()
        df = df.dropna(subset=[self.target_column])

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        if self.cat_cols is None:
            cat_cols = X.select_dtypes(exclude="number").columns.tolist()
        else:
            cat_cols = [c for c in self.cat_cols if c in X.columns]

        if self.num_impute and numeric_cols:
            for col in numeric_cols:
                if self.num_impute == "mean":
                    fill_value = X[col].mean()
                elif self.num_impute == "median":
                    fill_value = X[col].median()
                elif self.num_impute == "zero":
                    fill_value = 0
                else:
                    fill_value = None

                if fill_value is not None:
                    X[col] = X[col].fillna(fill_value)

        if self.cat_impute and cat_cols:
            for col in cat_cols:
                if self.cat_impute == "mode":
                    if X[col].mode().empty:
                        fill_value = self.cat_constant
                    else:
                        fill_value = X[col].mode().iloc[0]
                elif self.cat_impute == "constant":
                    fill_value = self.cat_constant
                else:
                    fill_value = None

                if fill_value is not None:
                    X[col] = X[col].fillna(fill_value)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        scale_cols = [c for c in self.scale_cols if c in X_train.columns]
        if scale_cols:
            X_train = X_train.copy()
            X_test = X_test.copy()
            X_train[scale_cols] = self.scaler.fit_transform(X_train[scale_cols])
            X_test[scale_cols] = self.scaler.transform(X_test[scale_cols])

        if cat_cols:
            X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
            X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        return X_train, X_test, y_train, y_test
