import pandas as pd
from pyspark.sql import SQLContext
from pyspark.sql.functions import col

import models.models_train as models
from preprocessors import functions as fun
from models.Kmeans import KmeansModel
from datetime import datetime as DateTimeFormatter
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

MIN_CLUSTER_DISTANCE = 0.5

DATA2015 = "E:\\diplom\predictormodel\\static\\yellow_tripdata_2015-01.csv"
DATA2016 = "E:\\diplom\predictormodel\\static\\yellow_tripdata_2016-01.csv"


def configure_spark():
    import os
    os.environ[
        'PYSPARK_SUBMIT_ARGS'] = '--driver-memory 4g ' \
                                 'pyspark-shell '


def load_data_spark():
    configure_spark()

    data_2015_uri = 'hdfs://localhost:9000/csv/yellow_tripdata_2015-01.csv'
    data_2016_uri = 'hdfs://localhost:9000/csv/yellow_tripdata_2016-01.csv'

    spark = SparkSession.builder.appName('abc').master('local').getOrCreate()

    data_2015 = spark.read.csv(data_2015_uri, header=True)
    data_2016 = spark.read.csv(data_2016_uri, header=True)

    # def filter_condition(x, year):
    #     return DateTimeFormatter.strptime(x['tpep_pickup_datetime'], "%Y-%m-%d %H:%M:%S").year == year

    # data_2015 = df.rdd.filter(lambda x: filter_condition(x, 2015))
    # print("2015: ", data_2015.count())
    #
    # data_2016 = df.rdd.filter(lambda x: filter_condition(x, 2016))
    # print("2016: ", data_2016.count())

    return data_2015, data_2016


def init_preprocessing(xg, k_means_model):
    print("init_preprocessing() - started")
    data_2015, data_2016 = load_data_spark()

    new_frame_cleaned = fun.preprocess(data_2015)
    new_frame_cleaned2 = fun.preprocess(data_2016)


    coord = new_frame_cleaned[["pickup_latitude", "pickup_longitude"]].values
    n_clusters = fun.pick_clusters_count(coord, MIN_CLUSTER_DISTANCE)
    coord = new_frame_cleaned[["pickup_latitude", "pickup_longitude"]].values

    new_frame_cleaned["pickup_cluster"] = k_means_model.train(coord, n_clusters) \
        .predict(new_frame_cleaned[["pickup_latitude", "pickup_longitude"]])

    print("init_preprocessing() - grouping by pickup_cluster", "time_bin")

    new_frame_cleaned2["pickup_cluster"] = k_means_model.predict(
        new_frame_cleaned2[["pickup_latitude", "pickup_longitude"]])

    jan_2016_data = fun.pickup_10min_bins(new_frame_cleaned2, 1, 2016)

    jan_2016_timeBin_groupBy = jan_2016_data[["pickup_cluster", "time_bin", "trip_distance"]].groupby(
        by=["pickup_cluster", "time_bin"]).count()

    unique_binswithPickup_Jan_2016 = fun.getUniqueBinsWithPickups(jan_2016_data, n_clusters)

    jan_2016_fillZero = fun.fillMissingWithZero(jan_2016_timeBin_groupBy["trip_distance"].values,
                                                unique_binswithPickup_Jan_2016, n_clusters)

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


def enrich_prediction_request(xg, k_means: KmeansModel, lat, lng, timestamp):
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
