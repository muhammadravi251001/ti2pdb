from pyspark.sql import functions as F
from pyspark.sql.functions import col,when,count
from pyspark.sql import SparkSession
import findspark
import pandas as pd

findspark.init()
spark = SparkSession.builder.master("local[*]").getOrCreate()

data = pd.read_csv("kredit.csv") 
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