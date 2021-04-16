import dask.dataframe as dd
import pandas as pd
import numpy as np
import models.models_train as models
from preprocessors import functions as fun
from models.Kmeans import KmeansModel

MIN_CLUSTER_DISTANCE = 0.5

DATA2015 = "E:\\diplom\predictormodel\\static\\yellow_tripdata_2015-01.csv"
DATA2016 = "E:\\diplom\predictormodel\\static\\yellow_tripdata_2016-01.csv"


def init_preprocessing(xg, k_means_model):
    print("init_preprocessing() - started")
    # data_2015 = dd.read_csv(DATA2015)
    # data_2016 = dd.read_csv(DATA2016)

    # new_frame_cleaned = preprocess(data_2015)
    # new_frame_cleaned2 = fun.preprocess(data_2016)
    new_frame_cleaned = pd.read_csv("E:\\diplom\\pythonProject\\static\\processed_2015_df.csv")
    new_frame_cleaned2 = pd.read_csv("E:\\diplom\\pythonProject\\static\\processed_2016_df.csv")

    coord = new_frame_cleaned[["pickup_latitude", "pickup_longitude"]].values
    # n_clusters = fun.pick_clusters_count(coord, MIN_CLUSTER_DISTANCE)
    n_clusters = 30
    coord = new_frame_cleaned[["pickup_latitude", "pickup_longitude"]].values

    new_frame_cleaned["pickup_cluster"] = k_means_model.train(coord, n_clusters) \
        .predict(new_frame_cleaned[["pickup_latitude", "pickup_longitude"]])

    jan_2015_data = fun.pickup_10min_bins(new_frame_cleaned, 1, 2015)
    print("init_preprocessing() - grouping by pickup_cluster", "time_bin")
    jan_2015_timeBin_groupBy = jan_2015_data[["pickup_cluster", "time_bin", "trip_distance"]].groupby(
        by=["pickup_cluster", "time_bin"]).count()

    new_frame_cleaned2["pickup_cluster"] = k_means_model.predict(
        new_frame_cleaned2[["pickup_latitude", "pickup_longitude"]])

    jan_2016_data = fun.pickup_10min_bins(new_frame_cleaned2, 1, 2016)

    jan_2016_timeBin_groupBy = jan_2016_data[["pickup_cluster", "time_bin", "trip_distance"]].groupby(
        by=["pickup_cluster", "time_bin"]).count()

    unique_binswithPickup_Jan_2015 = fun.getUniqueBinsWithPickups(jan_2015_data)

    jan_2015_fillSmooth = fun.smoothing(jan_2015_timeBin_groupBy["trip_distance"].values,
                                        unique_binswithPickup_Jan_2015)

    unique_binswithPickup_Jan_2016 = fun.getUniqueBinsWithPickups(jan_2016_data)

    jan_2016_fillZero = fun.fillMissingWithZero(jan_2016_timeBin_groupBy["trip_distance"].values,
                                                unique_binswithPickup_Jan_2016)

    regionWisePickup_Jan_2016 = []
    for i in range(30):
        regionWisePickup_Jan_2016.append(jan_2016_fillZero[4464 * i:((4464 * i) + 4464)])

    TruePickups, lat, lon, day_of_week, feat = fun.compute_pickups(k_means_model,
                                                                   regionWisePickup_Jan_2016)

    predicted_pickup_values_list = fun.compute_predicted_pickup_values(regionWisePickup_Jan_2016)

    train_df, train_TruePickups_flat, test_df, test_TruePickups_flat = fun.train_test_split_compute(
        regionWisePickup_Jan_2016, lat, lon, day_of_week, predicted_pickup_values_list,
        TruePickups, feat)

    models.xgboost_reg(train_df, train_TruePickups_flat, test_df, test_TruePickups_flat, xg)

    print(new_frame_cleaned2.loc[[1]][["pickup_latitude", "pickup_longitude", "pickup_cluster"]])
    print(new_frame_cleaned2.loc[[2]][["pickup_latitude", "pickup_longitude", "pickup_cluster"]])
    print(new_frame_cleaned2.loc[[1000]][["pickup_latitude", "pickup_longitude", "pickup_cluster"]])
    print(new_frame_cleaned2.loc[[1141249]][["pickup_latitude", "pickup_longitude", "pickup_cluster"]])