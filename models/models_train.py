from pandas import DataFrame
from pyspark.sql import SparkSession
import models.GbtModel as gbt


def xgboost_validate(train_X: DataFrame, train_y, test_X, test_y, ss: SparkSession):
    import pandas as pd

    def vectorize(cols, coords):
        from pyspark.ml.feature import VectorAssembler

        vectorAss = VectorAssembler(inputCols=cols, outputCol="features")
        vectorized_coords = vectorAss.transform(coords)
        return vectorized_coords

    features_cols = train_X.columns.tolist()

    train_X["target"] = pd.Series(train_y, index=train_X.index)
    test_X["target"] = pd.Series(test_y, index=test_X.index)

    train_df_spark = ss.createDataFrame(train_X)
    test_df_spark = ss.createDataFrame(test_X)

    vectorized_train = vectorize(features_cols, train_df_spark)
    vectorized_test = vectorize(features_cols, test_df_spark)

    model = gbt.GBTModelCustom(False)
    test_pred = model.cv().train(vectorized_train).validate(vectorized_test)
