from pyspark.ml.clustering import KMeans, KMeansModel

from pyspark.sql import SparkSession
from datetime import datetime

from preprocessors import HDFS_HOST


class KMeansModelCustom:

    def __init__(self, use_pretrained):
        self.data = None
        self.hdfs_uri = HDFS_HOST + "models/trained/kmeans/{}".format(datetime.now().date())
        self.sc = SparkSession.getActiveSession()
        if use_pretrained:
            self.model: KMeansModel = self.__load_from_hdfs()
        else:
            self.model:KMeansModel = None

    def train(self, data, n_clusters, save=False):
        if self.model is not None:
            raise RuntimeError("This model already trained! Create new instance and set pretrained flag as False")
        self.data = data
        print("k-means() - training")
        kmeans = KMeans(k=n_clusters)
        self.model = kmeans.fit(self.data)
        print("k-means() - training finished")
        if save:
            self.__save_to_hdfs__()
        return self

    def __save_to_hdfs__(self):
        self.model.write().overwrite().save(self.hdfs_uri)
        print("k-means() - model saved by uri {}".format(self.hdfs_uri))

    def __load_from_hdfs(self):
        sameModel = KMeansModel.load(self.hdfs_uri)
        print("k-means() - model loaded from uri {}".format(self.hdfs_uri))
        return sameModel

    def predict(self, to_predict):
        print("k-means() - discovering cluster")
        self.model.setPredictionCol("pickup_cluster")
        return self.model.transform(to_predict)

    def get_centers(self):
        return self.model.clusterCenters()

    def find_center(self, cluster_number):
        centers = self.model.clusterCenters()

    def __is_initialized__(self):
        if self.model is None:
            raise SystemError(
                "Kmeans model requires initialization. Call train() method to initialize model!")
