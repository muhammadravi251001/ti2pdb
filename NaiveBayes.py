from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder

sc = SparkContext('local')
spark = SparkSession(sc)

# Load training data
data = spark.read.format("libsvm") \
    .load("kredit.csv")

x_oh = OneHotEncoder(sparse=False)
x_oh = x_oh.fit_transform(data[['OCCUPATION']])
df_x_oh = DataFrame(x_oh)
df_x_oh

y_oh = OneHotEncoder(sparse=False)
y_oh = y_oh.fit_transform(data[[' MERK']])
df_y_oh = DataFrame(y_oh)
df_y_oh

concatenated = concat([data, df_x_oh, df_y_oh], axis="columns")

data_rev = concatenated.drop(["OCCUPATION"], axis = 1)
data_rev_1 = data_rev.drop([" MERK"], axis = 1)

# Split the data into train and test
splits = data_rev_1.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show()

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))