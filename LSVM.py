from pyspark.ml.classification import LinearSVC
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

if __name__ == "__main__":

    sc = SparkContext('local')
    spark = SparkSession(sc)

    # Load training data
    data = spark.read.format("libsvm") \
        .load("kredit.csv")

    categorical_columns= ['OCCUPATION', ' MERK']

    # The index of string vlaues multiple columns
    indexers = [
        StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
        for c in categorical_columns
    ]

    # The encode of indexed vlaues multiple columns
    encoders = [OneHotEncoder(dropLast=False,inputCol=indexer.getOutputCol(),
                outputCol="{0}_encoded".format(indexer.getOutputCol())) 
        for indexer in indexers
    ]

    # Vectorizing encoded values
    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders],outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders+[assembler])
    model=pipeline.fit(data)
    transformed = model.transform(data)

    # Split the data into train and test
    splits = transformed.randomSplit([0.6, 0.4], 1234)
    train = splits[0]
    test = splits[1]

    # create the trainer and set its parameters
    lsvc = LinearSVC(maxIter=10, regParam=0.1)

    # train the model
    model = lsvc.fit(train)

    # select example rows to display.
    predictions = model.transform(test)
    predictions.show()

    # compute accuracy on the test set
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test set accuracy = " + str(accuracy))

    spark.stop()