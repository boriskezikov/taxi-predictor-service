import pandas as pd
from pyspark.sql import DataFrame

from preprocessors import preprocessing_utils as fun, RAW_DATA_DRIVE, configure_spark, FINAL_PROCESSED, spark
from models.Kmeans import KMeansModelCustom
from pandas import DataFrame
from pyspark.sql import SparkSession
import models.GbtModel as gbt
from preprocessors.preprocessing_utils import vectorize



def load_data_spark():

    data_2015_uri: str = RAW_DATA_DRIVE + 'yellow_tripdata_2015-01.csv'
    data_2016_uri: str = RAW_DATA_DRIVE + 'yellow_tripdata_2016-01.csv'

    data_2015: DataFrame = spark.read.csv(data_2015_uri, header=True)
    data_2016: DataFrame = spark.read.csv(data_2016_uri, header=True)

    return data_2015, data_2016


def start_processing(n_clusters=30):
    print("init_preprocessing() - started")
    data_2015, data_2016 = load_data_spark()

    new_frame_cleaned: DataFrame = fun.preprocess(data_2015)
    new_frame_cleaned2: DataFrame = fun.preprocess(data_2016)

    coord = new_frame_cleaned[["pickup_latitude", "pickup_longitude"]]
    # n_clusters = fun.pick_clusters_count(coord, MIN_CLUSTER_DISTANCE)

    k_means = KMeansModelCustom(use_pretrained=False)
    vectorized = fun.coords_to_vector(coord)

    # training k-means on 2015-01
    k_means.train(vectorized, n_clusters, save=True)

    pickup_clusters_2016 = k_means.predict(
        fun.coords_to_vector(new_frame_cleaned2[["pickup_latitude", "pickup_longitude"]]))

    pickup_clusters_2016 = pickup_clusters_2016.drop("features")
    new_frame_cleaned2 = new_frame_cleaned2.join(pickup_clusters_2016, on=["pickup_latitude", "pickup_longitude"])
    print("init_preprocessing() - grouping by pickup_cluster", "time_bin")

    # todo добавить динамическое вычисление параметра года
    jan_2016_data = fun.pickup_10min_bins(new_frame_cleaned2, 1, 2016)

    jan_2016_timeBin_groupBy: DataFrame = jan_2016_data.select(["pickup_cluster", "time_bin", "trip_distance"]).groupBy(
        "pickup_cluster", "time_bin").count()

    jan_2016_fillZero = fun.fill_missing_tbins_with_zero(jan_2016_timeBin_groupBy, n_clusters)

    true_pickups, lat, lon, day_of_week, feat = fun.compute_pickups(jan_2016_fillZero, n_clusters)

    predicted_pickup_values_list = fun.compute_weighted_moving_average(jan_2016_fillZero, n_clusters)

    train_df, train_true_pickups_flat, test_df, test_true_pickups_flat = fun.train_test_split_compute(
        jan_2016_fillZero, lat, lon, day_of_week, predicted_pickup_values_list,
        true_pickups, feat, n_clusters)

    xgboost_validate(train_df, train_true_pickups_flat, test_df, test_true_pickups_flat, spark)


def xgboost_validate(train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame, test_y: pd.DataFrame,
                     ss: SparkSession):
    import pandas as pd

    features_cols = train_x.columns.tolist()

    # merging x,y into 1 dataset
    train_x["target"] = pd.Series(train_y, index=train_x.index)
    test_x["target"] = pd.Series(test_y, index=test_x.index)

    # converting pandas df to spark df
    train_df_spark = ss.createDataFrame(train_x)
    test_df_spark = ss.createDataFrame(test_x)

    train_df_spark.union(test_df_spark).write.csv(FINAL_PROCESSED, header=True, mode="overwrite")

    vectorized_train = vectorize(features_cols, train_df_spark)
    vectorized_test = vectorize(features_cols, test_df_spark)

    model = gbt.GBTModelCustom(False)
    model.cv().train(vectorized_train).validate(vectorized_test)
