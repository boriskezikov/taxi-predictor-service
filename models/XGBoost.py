from datetime import datetime

from xgboost import XGBRegressor
from pyspark.sql import SparkSession
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import joblib

from preprocessors import HDFS_HOST


class XGBoostModelCustom:

    def __init__(self, use_pretrained):
        self.train_x = None
        self.train_y = None
        self.hdfs_uri = HDFS_HOST + "models/trained/xgb-regressor/{}".format(datetime.now().date())
        self.sc = SparkSession.getActiveSession()
        if use_pretrained:
            self.model: XGBRegressor = self.__load_from_hdfs()
        else:
            self.model: XGBRegressor = None

    def cv(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

        print("xgboost() - starting cv")
        hyper_parameter = {"max_depth": [1, 2, 3, 4], "n_estimators": [40, 80, 150, 600]}
        clf: XGBRegressor = XGBRegressor()
        best_parameter = GridSearchCV(clf, hyper_parameter, scoring="neg_mean_absolute_error", cv=3)
        best_parameter.fit(self.train_x, self.train_y)
        estimators = best_parameter.best_params_["n_estimators"]
        depth = best_parameter.best_params_["max_depth"]
        self.model: XGBRegressor = XGBRegressor(max_depth=depth, n_estimators=estimators)
        print("xgboost() - cv finished. Model initialized")
        return self

    def train(self, save=False):
        self.__is_initialized__()
        print("xgboost() - training...")
        self.model.fit(self.train_x, self.train_y)
        train_pred = self.model.predict(self.train_x)
        train_MAPE = mean_absolute_error(self.train_y, train_pred) / (sum(self.train_y) / len(self.train_y))
        train_MSE = mean_squared_error(self.train_y, train_pred)
        print("xgboost() - training finished")
        print("Train MAPE: ", train_MAPE)
        print("Train MSE: ", train_MSE)
        if save:
            self.__save_to_hdfs__()
        return self

    def predict(self, to_predict):
        print("xgboost() - predicting.. ")
        return self.model.predict(to_predict)

    def __save_to_hdfs__(self):
        m:XGBRegressor = self.model
        m.save_model(self.hdfs_uri)
        # self.model.write().overwrite().save(self.hdfs_uri)
        print("xgboost() - model saved by uri {}".format(self.hdfs_uri))

    def __load_from_hdfs(self):
        sameModel =self.model.load(self.hdfs_uri)
        print("xgboost() - model loaded from uri {}".format(self.hdfs_uri))
        return sameModel

    def __is_initialized__(self):
        if self.model is None:
            raise SystemError(
                "XGBoost model requires initialization and cv boosting. Call cv() method to initialize model!")
