import numpy as np
import gpxpy.geo
from datetime import datetime
import time
import math
from sklearn.cluster import MiniBatchKMeans


def time_to_unix(t):
    change = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")  # this will convert the String time into datetime format
    t_tuple = change.timetuple()  # this will convert the datetime formatted time into structured time
    return time.mktime(t_tuple) + 3600  # this will convert structured time into unix-time.


# removes all pickup/dropoff points which remains out of current city
def clean_points_out_of_area(df):
    df = df[(
            ((df.pickup_latitude >= 40.5774) & (df.pickup_latitude <= 40.9176)) & (
            (df.pickup_longitude >= -74.15) & (df.pickup_longitude <= -73.7004)))]

    df = df[(
            ((df.dropoff_latitude >= 40.5774) & (df.dropoff_latitude <= 40.9176)) & (
            (df.dropoff_longitude >= -74.15) & (df.dropoff_longitude <= -73.7004)))]
    return df


def clean_points_by_incorrect_speed(df):
    df_cleaned = df[(df.trip_duration > 1) & (df.trip_duration < 720)]
    df_cleaned = df_cleaned[(df_cleaned.speed > 0) & (df_cleaned.speed < 45.31)]
    df_cleaned = df_cleaned[
        (df_cleaned.trip_distance > 0) & (df_cleaned.trip_distance < 23)]
    df_cleaned = df_cleaned[
        (df_cleaned.total_amount > 0) & (df_cleaned.total_amount < 86.6)]
    return df_cleaned


def df_with_trip_times(df):
    startTime = datetime.now()
    duration = df[["tpep_pickup_datetime", "tpep_dropoff_datetime"]].compute()
    pickup_time = [time_to_unix(pkup) for pkup in duration["tpep_pickup_datetime"].values]
    dropoff_time = [time_to_unix(drpof) for drpof in duration["tpep_dropoff_datetime"].values]
    trip_duration = (np.array(dropoff_time) - np.array(pickup_time)) / float(60)  # trip duration in minutes

    NewFrame = df[['passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                   'dropoff_latitude', 'total_amount']].compute()
    NewFrame["trip_duration"] = trip_duration
    NewFrame["pickup_time"] = pickup_time
    NewFrame["speed"] = (NewFrame["trip_distance"] / NewFrame["trip_duration"]) * 60

    print("Time taken for creation of dataframe is {}".format(datetime.now() - startTime))
    return NewFrame


def min_distance(regionCenters, totalClusters):
    less_dist = []
    more_dist = []
    min_distance = 100000  # any big number can be given here
    for i in range(totalClusters):
        good_points = 0
        bad_points = 0
        for j in range(totalClusters):
            if j != i:
                distance = gpxpy.geo.haversine_distance(latitude_1=regionCenters[i][0], longitude_1=regionCenters[i][1],
                                                        latitude_2=regionCenters[j][0], longitude_2=regionCenters[j][1])
                # you can check the documentation of above "gpxpy.geo.haversine_distance" at "https://github.com/tkrajina/gpxpy/blob/master/gpxpy/geo.py"
                # "gpxpy.geo.haversine_distance" gives distance between two latitudes and longitudes in meters. So, we have to convert it into miles.
                distance = distance / (1.60934 * 1000)  # distance from meters to miles
                min_distance = min(min_distance, distance)  # it will return minimum of "min_distance, distance".
                if distance < 2:
                    good_points += 1
                else:
                    bad_points += 1
        less_dist.append(good_points)
        more_dist.append(bad_points)
    print("On choosing a cluster size of {}".format(totalClusters))
    print("Avg. Number clusters within vicinity where inter cluster distance < 2 miles is {}".format(
        np.ceil(sum(less_dist) / len(less_dist))))
    print("Avg. Number clusters outside of vicinity where inter cluster distance > 2 miles is {}".format(
        np.ceil(sum(more_dist) / len(more_dist))))
    print("Minimum distance between any two clusters = {}".format(min_distance))
    print("-" * 10)
    return {totalClusters: min_distance}


def makingRegions(noOfRegions, coord):
    regions = MiniBatchKMeans(n_clusters=noOfRegions, batch_size=10000).fit(coord)
    regionCenters = regions.cluster_centers_
    totalClusters = len(regionCenters)
    return regionCenters, totalClusters


def changingLabels(num):
    if num < 10 ** 3:
        return num
    elif num >= 10 ** 3 and num < 10 ** 6:
        return str(num / 10 ** 3) + "k"
    elif num >= 10 ** 6 and num < 10 ** 9:
        return str(num / 10 ** 6) + "M"
    else:
        return str(num / 10 ** 9) + "B"


def pickup_10min_bins(dataframe, month, year):
    print("pickup_10min_bins() - Picking time bins")
    pickupTime = dataframe["pickup_time"].values
    unixTime = [1420070400, 1451606400]
    unix_year = unixTime[year - 2015]
    time_10min_bin = [int((i - unix_year) / 600) for i in pickupTime]
    dataframe["time_bin"] = np.array(time_10min_bin)
    print("pickup_10min_bins() - Picking time bins finished")
    return dataframe


def train_test_split_compute(regionWisePickup_Jan_2016, lat, lon, day_of_week, predicted_pickup_values_list,
                             TruePickups, feat, n_clusters):
    import pandas as pd
    print("train_test_split_compute() - started computing")

    amplitude_lists = []
    frequency_lists = []
    for i in range(n_clusters):
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

    train_previousFive_pickups = [feat[i * 4459:(4459 * i + 3567)] for i in range(n_clusters)]
    test_previousFive_pickups = [feat[(i * 4459) + 3567:(4459 * (i + 1))] for i in range(n_clusters)]
    train_fourier_frequencies = [frequency_lists[i * 4459:(4459 * i + 3567)] for i in range(n_clusters)]
    test_fourier_frequencies = [frequency_lists[(i * 4459) + 3567:(4459 * (i + 1))] for i in range(n_clusters)]
    train_fourier_amplitudes = [amplitude_lists[i * 4459:(4459 * i + 3567)] for i in range(n_clusters)]
    test_fourier_amplitudes = [amplitude_lists[(i * 4459) + 3567:(4459 * (i + 1))] for i in range(n_clusters)]

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
    for i in range(n_clusters):
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
    print("train_test_split_compute() - computing finished")
    return train_df, train_TruePickups_flat, test_df, test_TruePickups_flat


def compute_predicted_pickup_values(regionWisePickup_Jan_2016, n_clusters):
    print("compute_predicted_pickup_values() - started computing...")
    predicted_pickup_values = []
    predicted_pickup_values_list = []

    window_size = 2
    for i in range(n_clusters):
        for j in range(4464):
            if j == 0:
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
    print("compute_predicted_pickup_values() - computing finished")
    return predicted_pickup_values_list


def compute_pickups(k_means_model, regionWisePickup_Jan_2016, n_clusters):
    TruePickups = []
    lat = []
    lon = []
    day_of_week = []
    number_of_time_stamps = 5

    centerOfRegions = k_means_model.get_centers()
    feat = [0] * number_of_time_stamps
    for i in range(n_clusters):
        lat.append([centerOfRegions[i][0]] * 4459)
        lon.append([centerOfRegions[i][1]] * 4459)
        day_of_week.append([int(((int(j / 144) % 7) + 5) % 7) for j in range(5, 4464)])
        feat = np.vstack((feat, [regionWisePickup_Jan_2016[i][k:k + number_of_time_stamps] for k in
                                 range(0, len(regionWisePickup_Jan_2016[i]) - (number_of_time_stamps))]))
        TruePickups.append(regionWisePickup_Jan_2016[i][5:])
    feat = feat[1:]
    return TruePickups, lat, lon, day_of_week, feat


def getUniqueBinsWithPickups(dataframe, n_clusters):
    values = []
    for i in range(n_clusters):
        cluster_id = dataframe[dataframe["pickup_cluster"] == i]
        unique_clus_id = list(set(cluster_id["time_bin"]))
        unique_clus_id.sort()  # inplace sorting
        values.append(unique_clus_id)
    return values


def fillMissingWithZero(numberOfPickups, correspondingTimeBin, n_clusters):
    ind = 0
    smoothed_regions = []
    for c in range(0, n_clusters):
        smoothed_bins = []
        for t in range(4464):  # there are total 4464 time bins in both Jan-2015 & Feb-2016.
            if t in correspondingTimeBin[c]:  # if a time bin is present in "correspondingTimeBin" in cluster 'c',
                # then it means there is a pickup, in this case, we are simply adding number of pickups, else we are adding 0.
                smoothed_bins.append(numberOfPickups[ind])
                ind += 1
            else:
                smoothed_bins.append(0)
        smoothed_regions.extend(smoothed_bins)
    return smoothed_regions


def smoothing(numberOfPickups, correspondingTimeBin,n_clusters):
    ind = 0
    repeat = 0
    smoothed_region = []
    for cluster in range(0, n_clusters):
        smoothed_bin = []
        for t1 in range(4464):
            if repeat != 0:  # this will ensure that we shall not fill the pickup values again which we already filled by smoothing
                repeat -= 1
            else:
                if t1 in correspondingTimeBin[cluster]:
                    smoothed_bin.append(numberOfPickups[ind])
                    ind += 1
                else:
                    if t1 == 0:
                        # <---------------------CASE-1:Pickups missing in the beginning------------------------>
                        for t2 in range(t1, 4464):
                            if t2 not in correspondingTimeBin[cluster]:
                                continue
                            else:
                                right_hand_limit = t2
                                smoothed_value = (numberOfPickups[ind] * 1.0) / ((right_hand_limit + 1) * 1.0)
                                for i in range(right_hand_limit + 1):
                                    smoothed_bin.append(math.ceil(smoothed_value))
                                ind += 1
                                repeat = right_hand_limit - t1

                    if t1 != 0:
                        right_hand_limit = 0
                        for t2 in range(t1, 4464):
                            if t2 not in correspondingTimeBin[cluster]:
                                continue
                            else:
                                right_hand_limit = t2
                                break
                        if right_hand_limit == 0:
                            # <---------------------CASE-2: Pickups MISSING IN THE END------------------------------>
                            smoothed_value = (numberOfPickups[ind - 1] * 1.0) / (((4464 - t1) + 1) * 1.0)
                            del smoothed_bin[-1]
                            for i in range((4464 - t1) + 1):
                                smoothed_bin.append(math.ceil(smoothed_value))
                            repeat = (4464 - t1) - 1
                        # <---------------------CASE-3: Pickups MISSING IN MIDDLE OF TWO VALUES---------------->
                        else:
                            smoothed_value = ((numberOfPickups[ind - 1] + numberOfPickups[ind]) * 1.0) / (
                                    ((right_hand_limit - t1) + 2) * 1.0)
                            del smoothed_bin[-1]
                            for i in range((right_hand_limit - t1) + 2):
                                smoothed_bin.append(math.ceil(smoothed_value))
                            ind += 1
                            repeat = right_hand_limit - t1
        smoothed_region.extend(smoothed_bin)
    return smoothed_region


def countZeros(num):
    count = 0
    for i in num:
        if i == 0:
            count += 1
    return count


def simple_moving_average_ratios(ratios, n_clusters):
    predicted_ratio = (ratios["Ratio"].values)[0]
    predicted_ratio_values = []
    predicted_pickup_values = []
    absolute_error = []
    squared_error = []
    window_size = 3
    for i in range(4464 * n_clusters):
        if i % 4464 == 0:
            predicted_ratio_values.append(0)
            predicted_pickup_values.append(0)
            absolute_error.append(0)
            squared_error.append(0)
        else:
            predicted_ratio_values.append(predicted_ratio)
            predicted_pickup_values.append(int(predicted_ratio_values[i] * ratios["Given"].values[i]))
            absolute_error.append(
                abs(int(predicted_ratio_values[i] * ratios["Given"].values[i]) - ratios["Prediction"].values[i]))

            error = math.pow(
                (int(predicted_ratio_values[i] * ratios["Given"].values[i]) - ratios["Prediction"].values[i]), 2)
            squared_error.append(error)

        if (i + 1) >= window_size:
            predicted_ratio = sum(ratios["Ratio"].values[(i + 1) - window_size:(i + 1)])
            predicted_ratio = predicted_ratio / window_size
        else:
            predicted_ratio = sum(ratios["Ratio"].values[0:(i + 1)])
            predicted_ratio = predicted_ratio / (i + 1)

    ratios["Simple_Moving_Average_Ratios_Pred"] = predicted_pickup_values
    ratios["Simple_Moving_Average_Ratios_AbsError"] = absolute_error
    mean_absolute_percentage_error = (sum(absolute_error) / len(absolute_error)) / (
            sum(ratios["Prediction"]) / len(ratios["Prediction"]))
    mean_sq_error = sum(squared_error) / len(squared_error)
    return ratios, mean_absolute_percentage_error, mean_sq_error


def simple_moving_average_predictions(ratios, n_clusters):
    predicted_pickup = (ratios["Prediction"].values)[0]
    predicted_pickup_values = []
    absolute_error = []
    squared_error = []
    window_size = 2
    for i in range(4464 * n_clusters):
        if i % 4464 == 0:
            predicted_pickup_values.append(0)
            absolute_error.append(0)
            squared_error.append(0)
        else:
            predicted_pickup_values.append(predicted_pickup)
            absolute_error.append(abs(predicted_pickup_values[i] - ratios["Prediction"].values[i]))

            error = math.pow((predicted_pickup_values[i] - ratios["Prediction"].values[i]), 2)
            squared_error.append(error)

        if (i + 1) >= window_size:
            predicted_pickup = sum(ratios["Prediction"].values[(i + 1) - window_size:(i + 1)])
            predicted_pickup = predicted_pickup / window_size
        else:
            predicted_pickup = sum(ratios["Prediction"].values[0:(i + 1)])
            predicted_pickup = predicted_pickup / (i + 1)

    ratios["Simple_Moving_Average_Predictions_Pred"] = predicted_pickup_values
    ratios["Simple_Moving_Average_Predictions_AbsError"] = absolute_error
    mean_absolute_percentage_error = (sum(absolute_error) / len(absolute_error)) / (
            sum(ratios["Prediction"]) / len(ratios["Prediction"]))
    mean_sq_error = sum(squared_error) / len(squared_error)
    return ratios, mean_absolute_percentage_error, mean_sq_error


def weighted_moving_average_ratios(ratios, n_clusters):
    predicted_ratio = (ratios["Ratio"].values)[0]
    predicted_ratio_values = []
    predicted_pickup_values = []
    absolute_error = []
    squared_error = []
    window_size = 4
    for i in range(4464 * n_clusters):
        if i % 4464 == 0:
            predicted_ratio_values.append(0)
            predicted_pickup_values.append(0)
            absolute_error.append(0)
            squared_error.append(0)
        else:
            predicted_ratio_values.append(predicted_ratio)
            predicted_pickup_values.append(int(predicted_ratio_values[i] * ratios["Given"].values[i]))
            absolute_error.append(
                abs(int(predicted_ratio_values[i] * ratios["Given"].values[i]) - ratios["Prediction"].values[i]))

            error = math.pow(
                (int(predicted_ratio_values[i] * ratios["Given"].values[i]) - ratios["Prediction"].values[i]), 2)
            squared_error.append(error)

        if (i + 1) >= window_size:
            sumOfRatios = 0
            sumOfWeights = 0
            for j in range(window_size, 0, -1):
                sumOfRatios = sumOfRatios + j * (ratios["Ratio"].values)[i - window_size + j]
                sumOfWeights = sumOfWeights + j
            predicted_ratio = sumOfRatios / sumOfWeights
        else:
            sumOfRatios = 0
            sumOfWeights = 0
            for j in range(i + 1, 0, -1):
                sumOfRatios = sumOfRatios + j * (ratios["Ratio"].values)[j - 1]
                sumOfWeights = sumOfWeights + j
            predicted_ratio = sumOfRatios / sumOfWeights

    ratios["Weighted_Moving_Average_Ratios_Pred"] = predicted_pickup_values
    ratios["Weighted_Moving_Average_Ratios_AbsError"] = absolute_error
    mean_absolute_percentage_error = (sum(absolute_error) / len(absolute_error)) / (
            sum(ratios["Prediction"]) / len(ratios["Prediction"]))
    mean_sq_error = sum(squared_error) / len(squared_error)
    return ratios, mean_absolute_percentage_error, mean_sq_error


def weighted_moving_average_predictions(ratios, n_clusters):
    predicted_pickup = (ratios["Prediction"].values)[0]
    predicted_pickup_values = []
    absolute_error = []
    squared_error = []
    window_size = 2
    for i in range(4464 * n_clusters):
        if i % 4464 == 0:
            predicted_pickup_values.append(0)
            absolute_error.append(0)
            squared_error.append(0)
        else:
            predicted_pickup_values.append(predicted_pickup)
            absolute_error.append(abs(predicted_pickup_values[i] - ratios["Prediction"].values[i]))

            error = math.pow(int(predicted_pickup_values[i] - ratios["Prediction"].values[i]), 2)
            squared_error.append(error)

        if (i + 1) >= window_size:
            sumPickups = 0
            sumOfWeights = 0
            for j in range(window_size, 0, -1):
                sumPickups = sumPickups + j * (ratios["Prediction"].values)[i - window_size + j]
                sumOfWeights = sumOfWeights + j
            predicted_pickup = sumPickups / sumOfWeights
        else:
            sumPickups = 0
            sumOfWeights = 0
            for j in range(i + 1, 0, -1):
                sumPickups += j * (ratios["Prediction"].values)[j - 1]
                sumOfWeights += j
            predicted_pickup = sumPickups / sumOfWeights

    ratios["Weighted_Moving_Average_Predictions_Pred"] = predicted_pickup_values
    ratios["Weighted_Moving_Average_Predictions_AbsError"] = absolute_error
    mean_absolute_percentage_error = (sum(absolute_error) / len(absolute_error)) / (
            sum(ratios["Prediction"]) / len(ratios["Prediction"]))
    mean_sq_error = sum(squared_error) / len(squared_error)
    return ratios, mean_absolute_percentage_error, mean_sq_error


def exponential_weighted_moving_average_ratios(ratios, n_clusters):
    predicted_ratio = (ratios["Ratio"].values)[0]
    predicted_ratio_values = []
    predicted_pickup_values = []
    absolute_error = []
    squared_error = []
    alpha = 0.5
    for i in range(4464 * n_clusters):
        if i % 4464 == 0:
            predicted_ratio_values.append(0)
            predicted_pickup_values.append(0)
            absolute_error.append(0)
            squared_error.append(0)
        else:
            predicted_ratio_values.append(predicted_ratio)
            predicted_pickup_values.append(int(predicted_ratio_values[i] * ratios["Given"].values[i]))
            absolute_error.append(
                abs(int(predicted_ratio_values[i] * ratios["Given"].values[i]) - ratios["Prediction"].values[i]))
            error = math.pow(
                (int(predicted_ratio_values[i] * ratios["Given"].values[i]) - ratios["Prediction"].values[i]), 2)
            squared_error.append(error)
            predicted_ratio = alpha * predicted_ratio + ((1 - alpha) * ratios["Ratio"].values[i - 1])

    ratios["Exponential_Weighted_Moving_Average_Ratios_Pred"] = predicted_pickup_values
    ratios["Exponential_Weighted_Moving_Average_Ratios_AbsError"] = absolute_error
    mean_absolute_percentage_error = (sum(absolute_error) / len(absolute_error)) / (
            sum(ratios["Prediction"]) / len(ratios["Prediction"]))
    mean_sq_error = sum(squared_error) / len(squared_error)
    return ratios, mean_absolute_percentage_error, mean_sq_error


def exponential_weighted_moving_average_predictions(ratios, n_clusters):
    predicted_pickup = (ratios["Prediction"].values)[0]
    predicted_pickup_values = []
    absolute_error = []
    squared_error = []
    alpha = 0.5
    for i in range(4464 * n_clusters):
        if i % 4464 == 0:
            predicted_pickup_values.append(0)
            absolute_error.append(0)
            squared_error.append(0)
        else:
            predicted_pickup_values.append(predicted_pickup)
            absolute_error.append(abs(predicted_pickup_values[i] - ratios["Prediction"].values[i]))
            error = math.pow((predicted_pickup_values[i] - ratios["Prediction"].values[i]), 2)
            squared_error.append(error)
            predicted_pickup = alpha * predicted_pickup + ((1 - alpha) * ratios["Prediction"].values[i - 1])

    ratios["Exponential_Weighted_Moving_Average_Predictions_Pred"] = predicted_pickup_values
    ratios["Exponential_Weighted_Moving_Average_Predictions_AbsError"] = absolute_error
    mean_absolute_percentage_error = (sum(absolute_error) / len(absolute_error)) / (
            sum(ratios["Prediction"]) / len(ratios["Prediction"]))
    mean_sq_error = sum(squared_error) / len(squared_error)
    return ratios, mean_absolute_percentage_error, mean_sq_error


def preprocess(data):
    print("preprocess() - Starting baseline preprocessing")
    print("preprocess() - Calculating trip times")
    df = df_with_trip_times(data)
    print("preprocess() - Cleaning records with incorrect driver speed ")
    df = clean_points_by_incorrect_speed(df)
    print("preprocess() - Cleaning records out of area")
    df = clean_points_out_of_area(df)
    print("preprocess() - Finished baseline preprocessing")
    return df


def pick_clusters_count(coord, MIN_CLUSTER_DISTANCE):
    print("Beginning procedure of choosing clusters count..")
    startTime = datetime.now()
    clusters_min_dist = {}
    for i in range(10, 100, 10):
        regionCenters, totalClusters = makingRegions(i, coord)
        clusters_min_dist.update(min_distance(regionCenters, totalClusters))
    print("Finished procedure of choosing clusters count. Time taken = " + str(datetime.now() - startTime))
    for k in sorted(clusters_min_dist, key=clusters_min_dist.get, reverse=True):
        if clusters_min_dist[k] <= MIN_CLUSTER_DISTANCE:
            print("Appropriate clusters number: ", k)
            return k
