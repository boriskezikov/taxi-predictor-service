import dask.dataframe as dd
import pandas as pd
import numpy as np
import models.models_train as models
from preprocessors import functions as fun
from models.Kmeans import KmeansModel

MIN_CLUSTER_DISTANCE = 0.3

DATA2015 = "E:\\diplom\predictormodel\\static\\yellow_tripdata_2015-01.csv"
DATA2016 = "E:\\diplom\predictormodel\\static\\yellow_tripdata_2016-01.csv"


def init_preprocessing(xg):
    print("init_preprocessing() - started")
    # data_2015 = dd.read_csv(DATA2015)
    data_2016 = dd.read_csv(DATA2016)

    # new_frame_cleaned = preprocess(data_2015)
    new_frame_cleaned2 = fun.preprocess(data_2016)
    new_frame_cleaned2.to_csv("processed_2016_df.csv")
    print("Dumped")
    new_frame_cleaned = dd.read_csv("E:\\diplom\\pythonProject\\processed_2015_df.csv")

    coord = new_frame_cleaned[["pickup_latitude", "pickup_longitude"]].values
    n_clusters = fun.pick_clusters_count(coord, MIN_CLUSTER_DISTANCE)

    coord = new_frame_cleaned[["pickup_latitude", "pickup_longitude"]].values

    k_means_model = KmeansModel(coord, n_clusters)
    new_frame_cleaned["pickup_cluster"] = k_means_model.train() \
        .predict(new_frame_cleaned[["pickup_latitude", "pickup_longitude"]])

    jan_2015_data = fun.pickup_10min_bins(new_frame_cleaned, 1, 2015)
    jan_2015_timeBin_groupBy = jan_2015_data[["pickup_cluster", "time_bin", "trip_distance"]] \
        .groupby(by=["pickup_cluster", "time_bin"]).count()

    new_frame_cleaned2["pickup_cluster"] = k_means_model.predict(
        new_frame_cleaned2[["pickup_latitude", "pickup_longitude"]])
    jan_2016_data = fun.pickup_10min_bins(new_frame_cleaned2, 1, 2016)

    jan_2016_timeBin_groupBy = jan_2016_data[["pickup_cluster", "time_bin", "trip_distance"]].groupby(
        by=["pickup_cluster", "time_bin"]).count()

    unique_binswithPickup_Jan_2015 = fun.getUniqueBinsWithPickups(jan_2015_data)

    jan_2015_fillZero = fun.fillMissingWithZero(jan_2015_timeBin_groupBy["trip_distance"].values,
                                                unique_binswithPickup_Jan_2015)
    jan_2015_fillSmooth = fun.smoothing(jan_2015_timeBin_groupBy["trip_distance"].values,
                                        unique_binswithPickup_Jan_2015)

    unique_binswithPickup_Jan_2016 = fun.getUniqueBinsWithPickups(jan_2016_data)

    jan_2016_fillZero = fun.fillMissingWithZero(jan_2016_timeBin_groupBy["trip_distance"].values,
                                                unique_binswithPickup_Jan_2016)

    regionWisePickup_Jan_2016 = []
    for i in range(30):
        regionWisePickup_Jan_2016.append(jan_2016_fillZero[4464 * i:((4464 * i) + 4464)])

    Ratios_DF = pd.DataFrame()
    Ratios_DF["Given"] = jan_2015_fillSmooth
    Ratios_DF["Prediction"] = jan_2016_fillZero
    Ratios_DF["Ratio"] = Ratios_DF["Prediction"] * 1.0 / Ratios_DF["Given"] * 1.0

    print("Total Number of zeros in Ratio column = " + str(Ratios_DF["Ratio"].value_counts()[0]))
    print("Total Number of zeros in Prediction column = " + str(Ratios_DF["Prediction"].value_counts()[0]))

    r1, mape1, mse1 = fun.simple_moving_average_ratios(Ratios_DF)
    r2, mape2, mse2 = fun.simple_moving_average_predictions(Ratios_DF)
    r3, mape3, mse3 = fun.weighted_moving_average_ratios(Ratios_DF)
    r4, mape4, mse4 = fun.weighted_moving_average_predictions(Ratios_DF)
    r5, mape5, mse5 = fun.exponential_weighted_moving_average_ratios(Ratios_DF)
    r6, mape6, mse6 = fun.exponential_weighted_moving_average_predictions(Ratios_DF)

    number_of_time_stamps = 5

    TruePickups = []

    lat = []
    lon = []
    day_of_week = []

    centerOfRegions = k_means_model.get_centers()
    feat = [0] * number_of_time_stamps
    for i in range(30):
        lat.append([centerOfRegions[i][0]] * 4459)
        lon.append([centerOfRegions[i][1]] * 4459)
        day_of_week.append([int(((int(j / 144) % 7) + 5) % 7) for j in range(5, 4464)])
        feat = np.vstack((feat, [regionWisePickup_Jan_2016[i][k:k + number_of_time_stamps] for k in
                                 range(0, len(regionWisePickup_Jan_2016[i]) - (number_of_time_stamps))]))
        TruePickups.append(regionWisePickup_Jan_2016[i][5:])
    feat = feat[1:]
    predicted_pickup_values = []
    predicted_pickup_values_list = []

    predicted_value = -1

    window_size = 2
    for i in range(30):
        for j in range(4464):
            if j == 0:
                predicted_value = regionWisePickup_Jan_2016[i][j]
                predicted_pickup_values.append(0)
            else:
                if j >= window_size:
                    sumPickups = 0
                    sumOfWeights = 0
                    for k in range(window_size, 0, -1):
                        sumPickups += k * (regionWisePickup_Jan_2016[i][j - window_size + (k - 1)])
                        sumOfWeights += k
                    predicted_value = int(sumPickups / sumOfWeights)
                    predicted_pickup_values.append(predicted_value)
                else:
                    sumPickups = 0
                    sumOfWeights = 0
                    for k in range(j, 0, -1):
                        sumPickups += k * regionWisePickup_Jan_2016[i][k - 1]
                        sumOfWeights += k
                    predicted_value = int(sumPickups / sumOfWeights)
                    predicted_pickup_values.append(predicted_value)

        predicted_pickup_values_list.append(predicted_pickup_values[5:])
        predicted_pickup_values = []

    amplitude_lists = []
    frequency_lists = []
    for i in range(30):
        ampli = np.abs(np.fft.fft(regionWisePickup_Jan_2016[i][0:4096]))
        freq = np.abs(np.fft.fftfreq(4096, 1))
        ampli_indices = np.argsort(-ampli)[1:]
        amplitude_values = []
        frequency_values = []
        for j in range(0, 9, 2):
            amplitude_values.append(ampli[ampli_indices[j]])
            frequency_values.append(freq[ampli_indices[j]])
        for k in range(4459):
            amplitude_lists.append(amplitude_values)
            frequency_lists.append(frequency_values)

    print("size of total train data :" + str(int(133770 * 0.8)))
    print("size of total test data :" + str(int(133770 * 0.2)))
    print("size of train data for one cluster:" + str(int(4459 * 0.8)))
    print("size of total test data for one cluster:" + str(int(4459 * 0.2)))

    train_previousFive_pickups = [feat[i * 4459:(4459 * i + 3567)] for i in range(30)]
    test_previousFive_pickups = [feat[(i * 4459) + 3567:(4459 * (i + 1))] for i in range(30)]
    train_fourier_frequencies = [frequency_lists[i * 4459:(4459 * i + 3567)] for i in range(30)]
    test_fourier_frequencies = [frequency_lists[(i * 4459) + 3567:(4459 * (i + 1))] for i in range(30)]
    train_fourier_amplitudes = [amplitude_lists[i * 4459:(4459 * i + 3567)] for i in range(30)]
    test_fourier_amplitudes = [amplitude_lists[(i * 4459) + 3567:(4459 * (i + 1))] for i in range(30)]

    print(
        "Train Data: Total number of clusters = {}. Number of points in each cluster = {}. Total number of training points = {}".format(
            len(train_previousFive_pickups), len(train_previousFive_pickups[0]),
            len(train_previousFive_pickups) * len(train_previousFive_pickups[0])))
    print(
        "Test Data: Total number of clusters = {}. Number of points in each cluster = {}. Total number of test points = {}".format(
            len(test_previousFive_pickups), len(test_previousFive_pickups[0]),
            len(test_previousFive_pickups) * len(test_previousFive_pickups[0])))

    train_lat = [i[:3567] for i in lat]
    train_lon = [i[:3567] for i in lon]
    train_weekDay = [i[:3567] for i in day_of_week]
    train_weighted_avg = [i[:3567] for i in predicted_pickup_values_list]
    train_TruePickups = [i[:3567] for i in TruePickups]

    test_lat = [i[3567:] for i in lat]
    test_lon = [i[3567:] for i in lon]
    test_weekDay = [i[3567:] for i in day_of_week]
    test_weighted_avg = [i[3567:] for i in predicted_pickup_values_list]
    test_TruePickups = [i[3567:] for i in TruePickups]

    train_pickups = []
    test_pickups = []
    train_freq = []
    test_freq = []
    train_amp = []
    test_amp = []
    for i in range(30):
        train_pickups.extend(train_previousFive_pickups[i])
        test_pickups.extend(test_previousFive_pickups[i])
        train_freq.extend(train_fourier_frequencies[i])
        test_freq.extend(test_fourier_frequencies[i])
        train_amp.extend(train_fourier_amplitudes[i])
        test_amp.extend(test_fourier_amplitudes[i])

    train_prevPickups_freq_amp = np.hstack((train_pickups, train_freq, train_amp))
    test_prevPickups_freq_amp = np.hstack((test_pickups, test_freq, test_amp))

    train_flat_lat = sum(train_lat, [])
    train_flat_lon = sum(train_lon, [])
    train_flat_weekDay = sum(train_weekDay, [])
    train_weighted_avg_flat = sum(train_weighted_avg, [])
    train_TruePickups_flat = sum(train_TruePickups, [])

    test_flat_lat = sum(test_lat, [])
    test_flat_lon = sum(test_lon, [])
    test_flat_weekDay = sum(test_weekDay, [])
    test_weighted_avg_flat = sum(test_weighted_avg, [])
    test_TruePickups_flat = sum(test_TruePickups, [])

    columns = ['ft_5', 'ft_4', 'ft_3', 'ft_2', 'ft_1', 'freq1', 'freq2', 'freq3', 'freq4', 'freq5', 'Amp1', 'Amp2',
               'Amp3',
               'Amp4', 'Amp5']
    train_df = pd.DataFrame(data=train_prevPickups_freq_amp, columns=columns)
    train_df["Latitude"] = train_flat_lat
    train_df["Longitude"] = train_flat_lon
    train_df["WeekDay"] = train_flat_weekDay
    train_df["WeightedAvg"] = train_weighted_avg_flat

    test_df = pd.DataFrame(data=test_prevPickups_freq_amp, columns=columns)
    test_df["Latitude"] = test_flat_lat
    test_df["Longitude"] = test_flat_lon
    test_df["WeekDay"] = test_flat_weekDay
    test_df["WeightedAvg"] = test_weighted_avg_flat

    models.xgboost_reg(train_df, train_TruePickups_flat, test_df, test_TruePickups_flat, xg)
