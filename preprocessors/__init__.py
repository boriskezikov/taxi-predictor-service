from pyspark.sql import SparkSession
from datetime import datetime as dt

HDFS_HOST = "hdfs://localhost:9000"
PROCESSED_DATA_DRIVE = HDFS_HOST + "/{}/data/csv/processed/".format(dt.now().date())
RAW_DATA_DRIVE = HDFS_HOST + "/data/csv/raw/"
MIN_CLUSTER_DISTANCE = 0.5
FINAL_PROCESSED = PROCESSED_DATA_DRIVE + "processed_data_full.csv"

def configure_spark() -> SparkSession:
    import findspark
    import os
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-memory 8g ' \
                                        'pyspark-shell '

    findspark.init()
    spark = SparkSession.builder \
        .appName('taxi-predictor') \
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.executor.maxResultSize", "0") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
        .master('local[*]').getOrCreate()
    return spark

spark: SparkSession = configure_spark()
