package com.sample
import org.apache.spark.mllib.classification.{ NaiveBayes, NaiveBayesModel }
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession
object Sample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("ChurnPrediction")
      .getOrCreate()

    import spark.implicits._
    /* val ds = spark.read.text(args(0)).as[String]
    val counts =
      ds.flatMap(line => line.split(" "))
        .groupByKey(_.toLowerCase)
        .count()

    counts.show()*/

    val df = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("/spl/project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    val dfnNa = df.na.drop
    val processedDf2 = dfnNa.na.fill("")
    val processedDf = processedDf2.dropDuplicates()
    import org.apache.spark.sql.functions._

    val toInt = udf[Int, String](_.toInt)
    val toDouble = udf[Double, String](_.toDouble)
    val df1 = processedDf.withColumn("TotalCharges2", toDouble(col("TotalCharges")))
    val df2 = df1.withColumn("gender2", when(col("gender") === "Female", 0).otherwise(1))
    val df3 = df2.withColumn("partner2", when(col("partner") === "No", 0).otherwise(1))
    val df4 = df3.withColumn("dependents2", when(col("dependents") === "No", 0).otherwise(1))
    val df5 = df4.withColumn("contract2", when(col("contract") === "One year", 1).when(col("contract") === "Two year", 2).otherwise(0))
    val df6 = df5.withColumn("churn", when(col("churn") === "No", 0).otherwise(1))
    val dfFinal = df6.drop("customerID", "gender", "Partner", "Dependents", "tenure", "TotalCharges2", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "TotalCharges");

    val rdd = dfFinal.rdd

    val parsedData = rdd.map(
      row =>
        LabeledPoint(
          row.getInt(2),
          Vectors.dense(
            row.getInt(0),
            row.getDouble(1), row.getInt(3), row.getInt(4), row.getInt(5), row.getInt(6))))
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val trainData = splits(0);
    val model = NaiveBayes.train(splits(0), lambda = 1.0, modelType = "multinomial")
    print("\nNaive Bayes Prediction model has been built.\n")
    val predictionAndLabel = splits(1).map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / splits(1).count()
    print("##########################################################\naccuracy:" + accuracy * 100 + "\n")
  }
}
