import pandas as pd
from pyspark.sql import DataFrame

import models.models_train as models
from preprocessors import functions as fun
from models.Kmeans import KMeansModelCustom
from pyspark.sql import SparkSession

MIN_CLUSTER_DISTANCE = 0.5

DATA2015 = "E:\\diplom\predictormodel\\static\\yellow_tripdata_2015-01.csv"
DATA2016 = "E:\\diplom\predictormodel\\static\\yellow_tripdata_2016-01.csv"


def configure_spark() -> SparkSession:
    import os
    os.environ[
        'PYSPARK_SUBMIT_ARGS'] = '--driver-memory 8g ' \
                                 'pyspark-shell '
    spark = SparkSession.builder \
        .appName('abc') \
        .config("spark.driver.maxResultSize", "0") \
        .master('local[12]').getOrCreate()
    return spark


def load_data_spark():
    spark: SparkSession = configure_spark()

    data_2015_uri: str = 'hdfs://localhost:9000/raw_data/csv/test.csv'
    data_2016_uri: str = 'hdfs://localhost:9000/raw_data/csv/test.csv'

    data_2015: DataFrame = spark.read.csv(data_2015_uri, header=True)
    data_2016: DataFrame = spark.read.csv(data_2016_uri, header=True)

    return data_2015, data_2016


def init_preprocessing(xg):
    print("init_preprocessing() - started")
    data_2015, data_2016 = load_data_spark()

    new_frame_cleaned: DataFrame = fun.preprocess(data_2015)
    new_frame_cleaned2: DataFrame = fun.preprocess(data_2016)

    coord = new_frame_cleaned[["pickup_latitude", "pickup_longitude"]]
    n_clusters = fun.pick_clusters_count(coord, MIN_CLUSTER_DISTANCE, "hdfs://localhost:9000")
    coord = new_frame_cleaned[["pickup_latitude", "pickup_longitude"]]

    k_means = KMeansModelCustom("hdfs://localhost:9000", use_pretrained=False)
    vectorized = fun.coords_to_vector(coord)

    # training k-means on 2015-01
    k_means.train(vectorized, n_clusters, save=True).predict(
        fun.coords_to_vector(new_frame_cleaned[["pickup_latitude", "pickup_longitude"]]))

    # predicting cluster for 2016-01 using pretrained weights
    pickup_clusters_2016 = k_means.predict(
        fun.coords_to_vector(new_frame_cleaned2[["pickup_latitude", "pickup_longitude"]]))

    pickup_clusters_2016 = pickup_clusters_2016.drop("features")
    new_frame_cleaned2 = new_frame_cleaned2.join(pickup_clusters_2016, on=["pickup_latitude", "pickup_longitude"])
    print("init_preprocessing() - grouping by pickup_cluster", "time_bin")

    jan_2016_data = fun.pickup_10min_bins(new_frame_cleaned2, 1, 2016)

    # jan_2016_timeBin_groupBy = jan_2016_data[["pickup_cluster", "time_bin", "trip_distance"]].groupby(
    #     by=["pickup_cluster", "time_bin"]).count()

    jan_2016_timeBin_groupBy = jan_2016_data.select(["pickup_cluster", "time_bin", "trip_distance"]).groupBy(
        "pickup_cluster", "time_bin").count()

    unique_binswithPickup_Jan_2016: DataFrame = fun.getUniqueBinsWithPickups(jan_2016_data, n_clusters)


    # continue here
    jan_2016_fillZero = fun.fillMissingWithZero(jan_2016_timeBin_groupBy, unique_binswithPickup_Jan_2016, n_clusters)

    regionWisePickup_Jan_2016 = []
    for i in range(n_clusters):
        regionWisePickup_Jan_2016.append(jan_2016_fillZero[4464 * i:((4464 * i) + 4464)])

    TruePickups, lat, lon, day_of_week, feat = fun.compute_pickups(k_means_model, regionWisePickup_Jan_2016, n_clusters)

    predicted_pickup_values_list = fun.compute_predicted_pickup_values(regionWisePickup_Jan_2016, n_clusters)

    train_df, train_TruePickups_flat, test_df, test_TruePickups_flat = fun.train_test_split_compute(
        regionWisePickup_Jan_2016, lat, lon, day_of_week, predicted_pickup_values_list,
        TruePickups, feat, n_clusters)

    models.xgboost_reg(train_df, train_TruePickups_flat, test_df, test_TruePickups_flat, xg)

    print(new_frame_cleaned2.loc[[1]][["pickup_latitude", "pickup_longitude", "pickup_cluster"]])
    print(new_frame_cleaned2.loc[[2]][["pickup_latitude", "pickup_longitude", "pickup_cluster"]])
    print(new_frame_cleaned2.loc[[1000]][["pickup_latitude", "pickup_longitude", "pickup_cluster"]])
    print(new_frame_cleaned2.loc[[1141249]][["pickup_latitude", "pickup_longitude", "pickup_cluster"]])


def enrich_prediction_request(xg, k_means: KMeansModelCustom, lat, lng, timestamp):
    model_dto_df = pd.DataFrame(
        columns=["ft_5", "ft_4", "ft_3", "ft_2", "ft_1", "freq1", "freq2", "freq3", "freq4", "freq5", "Amp1", "Amp2",
                 "Amp3", "Amp4", "Amp5", "Latitude", "Longitude", "WeekDay", "WeightedAvg"])

    # f = np.array([lat, lng]).reshape((1, -1))
    # cluster = k_means.predict(f)
    model_dto_df._set_value(col="Latitude", value=lat, index=0)
    model_dto_df._set_value(col="Longitude", value=lng, index=0)
    model_dto_df._set_value(col="WeekDay", value=get_weekday(timestamp), index=0)
    xg.predict(model_dto_df)


def get_weekday(timestamp):
    return timestamp.now().weekday()
