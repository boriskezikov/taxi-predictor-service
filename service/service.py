import pandas as pd
import numpy as np
import preprocessors.preprocessing_utils as utils
from preprocessors.preprocessing_utils import vectorize

from preprocessors import FINAL_PROCESSED
from service import k_means, gbt, ss


def get_weekday(timestamp):
    return timestamp.now().weekday()


def enrich_prediction_request(lat, lng, n, timestamp):
    clusters = get_clusters(lat, lng, n)
    preprocessed_data = ss.read.csv(FINAL_PROCESSED, inferSchema=True, header=True).toPandas()
    to_predict = []
    for cluster in clusters:
        data4cluster: pd.DataFrame = preprocessed_data[preprocessed_data["Latitude"] == cluster[0]][
            preprocessed_data["Longitude"] == cluster[1]]
        ft_1 = data4cluster["ft_1"].mean()
        ft_2 = data4cluster["ft_2"].mean()
        ft_3 = data4cluster["ft_3"].mean()
        ft_4 = data4cluster["ft_4"].mean()
        ft_5 = data4cluster["ft_5"].mean()
        freq1 = data4cluster.iloc[1]["freq1"]
        freq2 = data4cluster.iloc[1]["freq2"]
        freq3 = data4cluster.iloc[1]["freq3"]
        freq4 = data4cluster.iloc[1]["freq4"]
        freq5 = data4cluster.iloc[1]["freq5"]
        amp1 = data4cluster.iloc[1]["Amp1"]
        amp2 = data4cluster.iloc[1]["Amp2"]
        amp3 = data4cluster.iloc[1]["Amp3"]
        amp4 = data4cluster.iloc[1]["Amp4"]
        amp5 = data4cluster.iloc[1]["Amp5"]
        wma = data4cluster["WeightedAvg"].mean()
        rec = create_record(ft_5, ft_4, ft_3, ft_2, ft_1, freq1, freq2, freq3, freq4, freq5, amp1, amp2, amp3, amp4,
                            amp5, cluster[0], cluster[1], get_weekday(timestamp), wma)
        to_predict.append(rec)
    to_predict_df = pd.concat(to_predict)
    df_spark = ss.createDataFrame(to_predict_df)
    vectorized = vectorize(df_spark.columns, df_spark)

    res: pd.DataFrame = gbt.predict(vectorized).toPandas()[["Latitude", "Longitude", "prediction"]]
    res: pd.DataFrame = res.sort_values("prediction").reset_index(drop=True)
    res["priority"] = res.index + 1
    res = res.drop("prediction", axis=1)
    return res.to_dict(orient='records')


def get_clusters(lat, lng, n):
    from sklearn.metrics import pairwise_distances_argmin_min

    requestor_location = np.array([float(lat), float(lng)]).reshape((1, -1))
    df = pd.DataFrame(data=requestor_location, columns=["lat", "lng"])
    df_Spark = ss.createDataFrame(df)
    df = utils.vectorize(df_Spark.columns, df_Spark)

    # current location cluster
    centers = k_means.get_centers()
    cluster = k_means.predict(df).collect()[0]["pickup_cluster"]
    current_cluster_coords = centers.pop(cluster)

    # find nearest n clusters around current
    nearest_centroids = pairwise_distances_argmin_min(centers, requestor_location)[1].tolist()
    nearest_centroids_dict = {}

    for c in range(len(nearest_centroids)):
        nearest_centroids_dict[c] = nearest_centroids[c]

    additional_clusters = sorted(nearest_centroids_dict, key=nearest_centroids_dict.get)[:int(n)]

    result = []

    for cluster_number in additional_clusters:
        result.append(centers[cluster_number])

    result.append(current_cluster_coords)

    return result


def create_record(ft_5, ft_4, ft_3, ft_2, ft_1, freq1, freq2, freq3, freq4, freq5, Amp1, Amp2, Amp3, Amp4, Amp5, Latitude, Longitude, WeekDay, WeightedAvg):
    data = [{"ft_5": ft_5, "ft_4": ft_4, "ft_3": ft_3, "ft_2": ft_2, "ft_1": ft_1, "freq1": freq1,
             "freq2": freq2, "freq3": freq3, "freq4": freq4, "freq5": freq5, "Amp1": Amp1, "Amp2": Amp2,
             "Amp3": Amp3, "Amp4": Amp4, "Amp5": Amp5, "Latitude": Latitude, "Longitude": Longitude,
             "WeekDay": WeekDay, "WeightedAvg": WeightedAvg}]
    model_dto_df = pd.DataFrame(data)
    return model_dto_df
