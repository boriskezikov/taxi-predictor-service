from pyspark.sql import SparkSession

HDFS_HOST = "hdfs://localhost:9000/"
PROCESSED_DATA_DRIVE = HDFS_HOST + "/data/csv/processed/"
RAW_DATA_DRIVE = HDFS_HOST + "/data/csv/raw/"


def configure_spark() -> SparkSession:
    import os
    os.environ[
        'PYSPARK_SUBMIT_ARGS'] = '--driver-memory 8g ' \
                                 'pyspark-shell '
    spark = SparkSession.builder \
        .appName('taxi-predictor') \
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.executor.maxResultSize", "0") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
        .master('local[*]').getOrCreate()

    return spark