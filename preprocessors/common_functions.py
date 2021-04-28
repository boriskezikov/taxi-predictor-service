import numpy as np
import gpxpy.geo
from datetime import datetime
import time
import math

import pyspark
from pyspark.rdd import PipelinedRDD
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import Row, DoubleType, ArrayType, StringType
from sklearn.cluster import MiniBatchKMeans
from models.Kmeans import KMeansModelCustom
from pyspark.sql.types import IntegerType

from preprocessors import HDFS_HOST


def time_to_unix(t) -> float:
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


def update_columns(df: DataFrame) -> DataFrame:
    # calculate trip duration and speed and add as columns to df
    def __add_duration_and_speed_cols(frame: DataFrame) -> DataFrame:
        frame: DataFrame = frame.withColumn("trip_duration", (frame["dropoff_time"] - frame["pickup_time"]) / float(60))
        frame: DataFrame = frame.withColumn("speed", (frame["trip_distance"] / frame["trip_duration"]) * 60)
        return frame

    from pyspark.sql.functions import unix_timestamp

    df: DataFrame = df.withColumn("pickup_time", unix_timestamp("tpep_pickup_datetime"))
    df: DataFrame = df.withColumn("dropoff_time", unix_timestamp("tpep_dropoff_datetime"))

    spark_df: DataFrame = __add_duration_and_speed_cols(df)

    cols_to_keep: list = ['passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude',
                          'dropoff_longitude',
                          'dropoff_latitude', 'total_amount', 'trip_duration', 'pickup_time', 'speed']

    # drop unnecessary columns
    spark_df: DataFrame = spark_df.drop(*(set(spark_df.columns) - set(cols_to_keep)))
    return spark_df


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


def coords_to_vector(lat_lng_df):
    def cast_to_double(lat_lng_df_inner: DataFrame):
        cols = lat_lng_df_inner.columns
        if len(cols) != 2:
            raise RuntimeError("Coordinated df must contain only 2 columns [lat, lng] but {0} found!".format(len(cols)))
        coord_doubles = lat_lng_df_inner.withColumn(cols[0], lat_lng_df_inner[0].cast(DoubleType())). \
            withColumn(cols[1], lat_lng_df_inner[1].cast(DoubleType()))
        return cols, coord_doubles

    def vectorize(cols, coords):
        from pyspark.ml.feature import VectorAssembler

        vectorAss = VectorAssembler(inputCols=cols, outputCol="features")
        vectorized_coords = vectorAss.transform(coords)
        return vectorized_coords

    cols, coord_doubles = cast_to_double(lat_lng_df)
    vectorized_coords = vectorize(cols, coord_doubles)
    return vectorized_coords


def makingRegions(noOfRegions: int, coord: DataFrame, hdfs_uri: str):
    vectorized_coords = coords_to_vector(coord)
    kmeans_model = KMeansModelCustom(use_pretrained=False)
    kmeans_model.train(vectorized_coords, n_clusters=noOfRegions)
    regionCenters = kmeans_model.get_centers()
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


def pickup_10min_bins(frame: DataFrame, month, year):
    print("pickup_10min_bins() - Picking time bins")
    unixTime = [1420070400, 1451606400]
    unix_year = unixTime[year - 2015]
    frame = frame.withColumn("time_bin", ((frame["pickup_time"] - unix_year) / 600).cast(
        IntegerType()))
    print("pickup_10min_bins() - Picking time bins finished")
    return frame


def train_test_split_compute(pickup_data_df:DataFrame, lat, lon, day_of_week, predicted_pickup_values_list,
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


def compute_predicted_pickup_values(pickup_data_df:DataFrame, n_clusters):
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


def compute_pickups(pickup_data_df:DataFrame, n_clusters):
    TruePickups = []
    lat = []
    lon = []
    day_of_week = []
    number_of_time_stamps = 5

    k_means_model = KMeansModelCustom(use_pretrained=True)
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


# возвращает список датафреймов, где индекс в списке == номеру кластера, а датафрейм содержит список уникальных временных бинов
def getUniqueBinsWithPickups(dataframe: DataFrame, n_clusters) -> list:
    values = []

    for i in range(n_clusters):
        cluster_id: DataFrame = dataframe[dataframe["pickup_cluster"] == i]
        unique_clus_id = cluster_id.select("time_bin").distinct().sort("time_bin", ascending=False)
        # unique_clus_id = list(set(cluster_id["time_bin"]))
        values.append(unique_clus_id)
    return values


def fill_missing_tbins_with_zero(pickup_bin_count_df: DataFrame, n_clusters):
    import pandas as pd

    now = datetime.now()
    print("fill_missing_tbins_with_zero() - starting..")
    ss = SparkSession.getActiveSession()
    print("fill_missing_tbins_with_zero() - caching data...")
    pickup_bin_count_df_pd: pd.DataFrame = pickup_bin_count_df.toPandas()
    print("fill_missing_tbins_with_zero() - caching finished")
    for cluster_id in range(0, n_clusters):
        now_for_cluster = datetime.now()

        current_cluster_df = pickup_bin_count_df_pd.loc[pickup_bin_count_df_pd.pickup_cluster == cluster_id]
        time_bins = current_cluster_df["time_bin"].unique()
        for time_bin in range(4464):  # todo добавить динамическое вычилсление количества бинов по месяцу
            #todo проверить совместимость типов str
            if time_bin not in time_bins:
                pickup_bin_count_df_pd = pickup_bin_count_df_pd.append({
                    "pickup_cluster": cluster_id,
                    "time_bin": time_bin,
                    "count": 0
                }, ignore_index=True)
        print("fill_missing_tbins_with_zero() - cluster {0} processing finished. time taken {1}".format(cluster_id,
                                                                                                        datetime.now() - now_for_cluster))
    pickup_bin_count_df_pd = pickup_bin_count_df_pd.loc[pickup_bin_count_df_pd.time_bin >= 0]
    print("fill_missing_tbins_with_zero() - time taken {}".format(datetime.now() - now))
    assert len(pickup_bin_count_df_pd.index) == 4464 * 30
    return ss.createDataFrame(pickup_bin_count_df_pd)


def fill_missing_tbins_with_zero_withoud_collecting(pickup_bin_count_df: DataFrame, n_clusters):
    now = datetime.now()
    print("fill_missing_tbins_with_zero() - starting..")
    ss = SparkSession.getActiveSession()
    print("fill_missing_tbins_with_zero() - caching data...")
    pickup_bin_count_df = pickup_bin_count_df.cache()
    print("fill_missing_tbins_with_zero() - caching finished")
    for cluster_id in range(0, n_clusters):
        print("fill_missing_tbins_with_zero() - processing cluster {0}. {1} - left".format(cluster_id,
                                                                                           n_clusters - cluster_id))
        for time_bin in range(4464):  # todo добавить динамическое вычилсление количества бинов по месяцу
            row = ss.createDataFrame([(cluster_id, time_bin, 0)], "pickup_cluster int, time_bin int, count int")
            pickup_bin_count_df = pickup_bin_count_df.union(row)

    from pyspark.sql.window import Window
    import pyspark.sql.functions as F
    from pyspark.sql.functions import col

    pickup_bin_count_df = pickup_bin_count_df.select("pickup_cluster", "time_bin", "count", F.row_number().over(
        Window.partitionBy("count").orderBy(pickup_bin_count_df['count'])).alias("row_num")).sort(col("count"))
    pickup_bin_count_df = pickup_bin_count_df.filter(pickup_bin_count_df.row_num == 1).show()

    print("fill_missing_tbins_with_zero() - time taken {}".format(datetime.now() - now))
    return pickup_bin_count_df


def smoothing(numberOfPickups, correspondingTimeBin, n_clusters):
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


def preprocess(data: DataFrame) -> DataFrame:
    print("preprocess() - Starting baseline preprocessing")
    print("preprocess() - Calculating trip times")
    df = update_columns(data)
    print("preprocess() - Cleaning records with incorrect driver speed ")
    df = clean_points_by_incorrect_speed(df)
    print("preprocess() - Cleaning records out of area")
    df: DataFrame = clean_points_out_of_area(df)
    print("preprocess() - Finished baseline preprocessing")
    return df


def pick_clusters_count(coord, MIN_CLUSTER_DISTANCE, hdfs_uri):
    print("Beginning procedure of choosing clusters count..")
    startTime = datetime.now()
    clusters_min_dist = {}
    for i in range(10, 100, 10):
        regionCenters, totalClusters = makingRegions(i, coord, hdfs_uri)
        clusters_min_dist.update(min_distance(regionCenters, totalClusters))
    print("Finished procedure of choosing clusters count. Time taken = " + str(datetime.now() - startTime))
    for k in sorted(clusters_min_dist, key=clusters_min_dist.get, reverse=True):
        if clusters_min_dist[k] <= MIN_CLUSTER_DISTANCE:
            print("Appropriate clusters number: ", k)
            return k
