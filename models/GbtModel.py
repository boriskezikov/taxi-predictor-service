from datetime import datetime
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.sql import SparkSession

from preprocessors import HDFS_HOST


class GBTModelCustom:

    def __init__(self, use_pretrained):
        self.train_data = None
        self.hdfs_uri = HDFS_HOST + "models/trained/gbt-regressor/{}".format(datetime.now().date())
        self.sc = SparkSession.getActiveSession()
        self.use_pretrained = use_pretrained
        if use_pretrained:
            self.model: GBTRegressor = self.__load_from_hdfs()
        else:
            self.model: GBTRegressor = GBTRegressor(featuresCol="features", maxIter=20, labelCol="target")

    def cv(self):
        print("GBT() - starting cv")
        if not self.use_pretrained:
            print("GBT() - cv finished. Model initialized")
        print("GBT() - cv stopped. Using pretrained model")
        return self

    def train(self, train_data):
        self.train_data = train_data
        self.__is_initialized_raise__()
        if not self.use_pretrained:
            print("GBT() - training...")
            self.model = self.model.fit(self.train_data)
            print("GBT() - training finished")
            print("GBT() - evaluation: {}")
            self.__save_to_hdfs__()
        print("GBT() - train skipped")
        return self

    def validate(self, val_data):
        print("GBT() - validating.. ")
        self.model.getPredictionCol()
        predictions = self.model.transform(val_data)
        mae, mse, rmse, r2, var = self.__evaluate__(predictions)
        print("MAE on test data = %g" % mae)
        print("MSE on test data = %g" % mse)
        print("RMSE on test data = %g" % rmse)
        print("R2 on test data = %g" % r2)
        print("VAR on test data = %g" % var)

    def predict(self, to_predict):
        print("GBT() - predicting.. ")
        self.model.getPredictionCol()
        predictions = self.model.transform(to_predict)
        return predictions.toPandas()["prediction"]

    def __save_to_hdfs__(self):
        self.model.write().overwrite().save(self.hdfs_uri)
        print("GBT() - model saved by uri {}".format(self.hdfs_uri))

    def __evaluate__(self, predictions):
        evaluator = RegressionEvaluator(labelCol="target")
        mae = evaluator.setMetricName("mae").evaluate(predictions)
        rmse = evaluator.setMetricName("rmse").evaluate(predictions)
        mse = evaluator.setMetricName("mse").evaluate(predictions)
        r2 = evaluator.setMetricName("r2").evaluate(predictions)
        var = evaluator.setMetricName("var").evaluate(predictions)
        return mae, mse, rmse, r2, var

    def __load_from_hdfs(self):
        mdl = GBTRegressor.load(self.hdfs_uri)
        print("GBT() - model loaded from uri {}".format(self.hdfs_uri))
        return mdl

    def __is_initialized_raise__(self):
        if self.model is None:
            raise SystemError(
                "GBT model requires initialization and cv boosting. Call cv() method to initialize model!")
