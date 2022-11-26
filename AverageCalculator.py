# Script task PySpark MapReduce - rata-rata SALARY

from pyspark.sql import functions as F
from pyspark.sql import SparkSession
import findspark
import pandas as pd

findspark.init()
spark = SparkSession.builder.master("local[*]").getOrCreate()

data = pd.read_csv("kredit.csv") 
df = spark.createDataFrame(data, ("OCCUPATION", "SALARY", "INSTALLMENT", "TENOR", "USIA", "MERK", "STATUS"))

columns = df.columns
columns.remove('OCCUPATION')
columns.remove('INSTALLMENT')
columns.remove('TENOR')
columns.remove('OCCUUSIAPATION')
columns.remove('MERK')
columns.remove('STATUS')

cols_to_agg = [f(c) for c in columns for f in [F.avg]]

df.agg(*cols_to_agg).show()