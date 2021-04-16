from sklearn.cluster import MiniBatchKMeans


class KmeansModel:

    def __init__(self):
        self.model = None
        self.data = None
        self.clusters = None

    def train(self, data, n_clusters):
        self.model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000)
        self.data = data
        print("k-means() - training")
        self.clusters = self.model.fit(self.data)
        print("k-means() - training finished")
        return self

    def predict(self, to_predict):
        print("k-means() - discovering cluster")
        return self.model.predict(to_predict)

    def get_centers(self):
        return self.model.cluster_centers_

    def find_center(self, cluster_number):
        centers = self.model.cluster_centers_

    def __is_initialized__(self):
        if self.model is None:
            raise SystemError(
                "Kmeans model requires initialization. Call train() method to initialize model!")