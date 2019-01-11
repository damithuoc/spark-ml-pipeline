package demo.spark.pipeline

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row


object SparkPipelineExample extends App {

  // Create Spark session
  val spark = SparkSession
    .builder()
    .appName("SparkPipelineExample") // App name it can be any name, here I used 'SparkPipelineExample'
    .master("local[*]") // Master node, here I used local
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR") // I set log level as "ERROR", this will easy to see output

  // Create a sample data set (data frame)
  val df = spark.createDataFrame(Seq(
    (0L, "a b c d e spark", 1.0), // columns => id, text, label
    (1L, "b d", 0.0),
    (2L, "spark f g h", 1.0),
    (3L, "hadoop mapreduce", 0.0)
  )).toDF("id", "text", "label")


  // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
  val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")

  val hashingTF = new HashingTF()
    .setNumFeatures(1000)
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("features")

  // Create a logistic regression
  val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.001)

  val stages = Array(tokenizer, hashingTF, lr)

  val pipeline = new Pipeline()
    .setStages(stages)

  // Fit the pipeline to training documents.
  val model = pipeline.fit(df)

  // Now we can optionally save the fitted pipeline to disk
  model.write.overwrite().save("/tmp/SparkPipelineModel")

  // We can also save this unfit pipeline to disk
  pipeline.write.overwrite().save("/tmp/unfit-lr-model")

  // And load it back in during production
  val sameModel = PipelineModel.load("/tmp/SparkPipelineModel")

  // Prepare test documents, which are unlabeled (id, text) tuples.
  val test = spark.createDataFrame(Seq(
    (4L, "spark i j k"),
    (5L, "l m n"),
    (6L, "spark hadoop spark"),
    (7L, "apache hadoop")
  )).toDF("id", "text")

  // Make predictions on test documents.
  model.transform(test)
    .select("id", "text", "probability", "prediction")
    .collect()
    .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
      println(s"($id, $text) --> prob=$prob, prediction=$prediction")
    }


  spark.stop()

}