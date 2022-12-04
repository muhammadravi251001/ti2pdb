# Script task PySpark MapReduce - rata-rata SALARY

from pyspark.sql import functions as F
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

if __name__ == "__main__":

    sc = SparkContext('local')
    spark = SparkSession(sc)

    data = spark.read.options(header=True, inferSchema=True).csv('kredit.csv')
    df = spark.createDataFrame(data)

    columns = df.columns
    columns.remove('OCCUPATION')
    columns.remove('INSTALLMENT')
    columns.remove('TENOR')
    columns.remove('OCCUUSIAPATION')
    columns.remove('MERK')
    columns.remove('STATUS')

    cols_to_agg = [f(c) for c in columns for f in [F.avg]]

    df.agg(*cols_to_agg).show()

    spark.stop()