import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import joblib


class XGBoostModel:

    def __init__(self):
        self.model = None
        self.train_x = None
        self.train_y = None

    def cv(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

        print("xgboost() - starting cv")
        hyper_parameter = {"max_depth": [1, 2, 3, 4], "n_estimators": [40, 80, 150, 600]}
        clf = xgb.XGBRegressor()
        best_parameter = GridSearchCV(clf, hyper_parameter, scoring="neg_mean_absolute_error", cv=3)
        best_parameter.fit(self.train_x, self.train_y)
        estimators = best_parameter.best_params_["n_estimators"]
        depth = best_parameter.best_params_["max_depth"]
        self.model = xgb.XGBRegressor(max_depth=depth, n_estimators=estimators)
        print("xgboost() - cv finished. Model initialized")
        return self

    def train(self):
        self.__is_initialized__()
        print("xgboost() - training...")
        self.model.fit(self.train_x, self.train_y)
        train_pred = self.model.predict(self.train_x)
        train_MAPE = mean_absolute_error(self.train_y, train_pred) / (sum(self.train_y) / len(self.train_y))
        train_MSE = mean_squared_error(self.train_y, train_pred)
        print("xgboost() - training finished")
        print("Train MAPE: ", train_MAPE)
        print("Train MSE: ", train_MSE)
        self.dump_model()
        return self

    def predict(self, to_predict):
        print("xgboost() - predicting.. ")
        return self.model.predict(to_predict)

    def dump_model(self):
        print("xgboost() - dumping model.. ")
        joblib.dump(self.model, "dumped_models/xgb-trained")
        print("xgboost() - dumping finished. ")

    def __is_initialized__(self):
        if self.model is None:
            raise SystemError(
                "XGBoost model requires initialization and cv boosting. Call cv() method to initialize model!")
