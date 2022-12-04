# Script task PySpark MapReduce - jumlah data STATUS

from pyspark.sql import functions as F
from pyspark.sql.functions import col,when,count
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

if __name__ == "__main__":

    sc = SparkContext('local')
    spark = SparkSession(sc)

    data = spark.read.csv("kredit.csv")
    df = spark.createDataFrame(data, ("OCCUPATION", "SALARY", "INSTALLMENT", "TENOR", "USIA", "MERK", "STATUS"))

    columns = df.columns
    columns.remove('OCCUPATION')
    columns.remove('SALARY')
    columns.remove('INSTALLMENT')
    columns.remove('TENOR')
    columns.remove('OCCUUSIAPATION')
    columns.remove('MERK')

    cols_to_agg_lunas = [f(c) for c in columns for f in [F.count(F.when(col("STATUS") == "LUNAS", True))]]
    cols_to_agg_tarikan = [f(c) for c in columns for f in [F.count(F.when(col("STATUS") == "TARIKAN", True))]]

    df.agg(*cols_to_agg_lunas).show()
    df.agg(*cols_to_agg_tarikan).show()

    spark.stop()