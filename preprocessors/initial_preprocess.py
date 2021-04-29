from datetime import datetime

import pandas as pd
from pyspark.sql import DataFrame

import models.models_train as models
from preprocessors import common_functions as fun, RAW_DATA_DRIVE, HDFS_HOST, PROCESSED_DATA_DRIVE, configure_spark
from models.Kmeans import KMeansModelCustom
from pyspark.sql import SparkSession

MIN_CLUSTER_DISTANCE = 0.5

spark: SparkSession = configure_spark()


def load_data_spark():
    data_2015_uri: str = RAW_DATA_DRIVE + 'yellow_tripdata_2015-01.csv'
    data_2016_uri: str = RAW_DATA_DRIVE + 'yellow_tripdata_2016-01.csv'

    data_2015: DataFrame = spark.read.csv(data_2015_uri, header=True)
    data_2016: DataFrame = spark.read.csv(data_2016_uri, header=True)

    return data_2015, data_2016


def init_preprocessing():
    # print("init_preprocessing() - started")
    # data_2015, data_2016 = load_data_spark()
    #
    # new_frame_cleaned: DataFrame = fun.preprocess(data_2015)
    # new_frame_cleaned2: DataFrame = fun.preprocess(data_2016)
    #
    # # temp for testing
    # new_frame_cleaned.write.csv(PROCESSED_DATA_DRIVE + "new_frame_cleaned.csv", header=True, mode="overwrite")
    # new_frame_cleaned2.write.csv(PROCESSED_DATA_DRIVE + "new_frame_cleaned2.csv", header=True, mode="overwrite")
    # new_frame_cleaned = spark.read.csv(PROCESSED_DATA_DRIVE + "new_frame_cleaned.csv", header=True)
    # new_frame_cleaned2 = spark.read.csv(PROCESSED_DATA_DRIVE + "new_frame_cleaned2.csv", header=True)
    #
    # coord = new_frame_cleaned[["pickup_latitude", "pickup_longitude"]]
    # # n_clusters = fun.pick_clusters_count(coord, MIN_CLUSTER_DISTANCE, "hdfs://localhost:9000")

    n_clusters = 30
    #
    # k_means = KMeansModelCustom(use_pretrained=False)
    # vectorized = fun.coords_to_vector(coord)
    # #
    # # training k-means on 2015-01
    # k_means.train(vectorized, n_clusters, save=True)
    #
    # # predicting cluster for 2016-01 using pretrained weights
    # pickup_clusters_2016 = k_means.predict(
    #     fun.coords_to_vector(new_frame_cleaned2[["pickup_latitude", "pickup_longitude"]]))
    #
    # pickup_clusters_2016 = pickup_clusters_2016.drop("features")
    # new_frame_cleaned2 = new_frame_cleaned2.join(pickup_clusters_2016, on=["pickup_latitude", "pickup_longitude"])
    # print("init_preprocessing() - grouping by pickup_cluster", "time_bin")
    #
    # jan_2016_data = fun.pickup_10min_bins(new_frame_cleaned2, 1,
    #                                       2016)  # todo добавить динамическое вычисление параметра года
    #
    # jan_2016_timeBin_groupBy: DataFrame = jan_2016_data.select(["pickup_cluster", "time_bin", "trip_distance"]).groupBy(
    #     "pickup_cluster", "time_bin").count()
    #
    # jan_2016_timeBin_groupBy.write.csv(PROCESSED_DATA_DRIVE + "jan_2016_timeBin_groupBy.csv", header=True,
    #                                    mode="overwrite")
    # jan_2016_timeBin_groupBy = spark.read.csv(PROCESSED_DATA_DRIVE + "jan_2016_timeBin_groupBy.csv", header=True,
    #                                           inferSchema=True)
    #
    # jan_2016_fillZero = fun.fill_missing_tbins_with_zero(jan_2016_timeBin_groupBy, n_clusters)
    # jan_2016_fillZero.write.csv(PROCESSED_DATA_DRIVE + "jan_2016_fillZero_{}.csv".format(datetime.now().date()), header=True,
    #     mode="overwrite")
    jan_2016_fillZero = spark.read.csv(
        PROCESSED_DATA_DRIVE + "jan_2016_fillZero_{}.csv".format(datetime.now().date()), header=True,
        inferSchema=True)

    TruePickups, lat, lon, day_of_week, feat = fun.compute_pickups(jan_2016_fillZero, n_clusters)

    predicted_pickup_values_list = fun.compute_weighted_moving_average(jan_2016_fillZero, n_clusters)

    train_df, train_TruePickups_flat, test_df, test_TruePickups_flat = fun.train_test_split_compute(
        jan_2016_fillZero, lat, lon, day_of_week, predicted_pickup_values_list,
        TruePickups, feat, n_clusters)

    models.xgboost_validate(train_df, train_TruePickups_flat, test_df, test_TruePickups_flat)


def enrich_prediction_request(xg, k_means: KMeansModelCustom, lat, lng, timestamp):
    def get_weekday(timestamp):
        return timestamp.now().weekday()

    model_dto_df = pd.DataFrame(
        columns=["ft_5", "ft_4", "ft_3", "ft_2", "ft_1", "freq1", "freq2", "freq3", "freq4", "freq5", "Amp1", "Amp2",
                 "Amp3", "Amp4", "Amp5", "Latitude", "Longitude", "WeekDay", "WeightedAvg"])

    # f = np.array([lat, lng]).reshape((1, -1))
    # cluster = k_means.predict(f)
    model_dto_df._set_value(col="Latitude", value=lat, index=0)
    model_dto_df._set_value(col="Longitude", value=lng, index=0)
    model_dto_df._set_value(col="WeekDay", value=get_weekday(timestamp), index=0)
    xg.predict(model_dto_df)
