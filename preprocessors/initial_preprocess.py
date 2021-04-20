import dask.dataframe as dd
import pandas as pd
import numpy as np
import models.models_train as models
from preprocessors import functions as fun
from models.Kmeans import KmeansModel
from datetime import datetime
import sqlalchemy
import psycopg2

MIN_CLUSTER_DISTANCE = 0.5

DATA2015 = "E:\\diplom\predictormodel\\static\\yellow_tripdata_2015-01.csv"
DATA2016 = "E:\\diplom\predictormodel\\static\\yellow_tripdata_2016-01.csv"


def load_2015_sql():
    print("sql start")
    username = 'adminapp'
    password = 'adminapp'
    hostname = 'localhost'
    port = '5432'
    database_name = 'taxi_db'
    connection_string = 'postgresql://{0}:{1}@{2}:{3}/{4}'.format(username, password, hostname, port, database_name)
    data = dd.read_sql_table(index_col="id", uri=connection_string, table="records")
    print("sql finished")
    return data


def init_preprocessing(xg, k_means_model):
    print("init_preprocessing() - started")
    # data_2015 = dd.read_csv(DATA2015)
    # data_2016 = dd.read_csv(DATA2016)
    #
    # new_frame_cleaned = fun.preprocess(data_2015)
    # new_frame_cleaned2 = fun.preprocess(data_2016)
    new_frame_cleaned = pd.read_csv("E:\\diplom\\pythonProject\\static\\processed_2015_df.csv")
    new_frame_cleaned2 = pd.read_csv("E:\\diplom\\pythonProject\\static\\processed_2016_df.csv")

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


def get_weekday(timestamp: datetime):
    return timestamp.now().weekday()
