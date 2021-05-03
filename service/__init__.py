from models.Kmeans import KMeansModelCustom
from models.GbtModel import  GBTModelCustom
from pyspark.sql import SparkSession

k_means = KMeansModelCustom(True)
gbt = GBTModelCustom(True)
ss = SparkSession.getActiveSession()
