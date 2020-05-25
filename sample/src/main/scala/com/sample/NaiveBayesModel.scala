package com.sample
import java.io.FileNotFoundException
import java.io.IOException

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

object NaiveBayesModel {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("ChurnPrediction")
      .getOrCreate()

    /* val ds = spark.read.text(args(0)).as[String]
    val counts =
      ds.flatMap(line => line.split(" "))
        .groupByKey(_.toLowerCase)
        .count()

    counts.show()*/
    try {
      val df = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("/spl/project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

      val rddinit = spark.sparkContext.textFile("/spl/project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
      val temprdd = rddinit.take(10)
      temprdd.foreach(line => println(line))
      //df.
      print("\nSchema of the initial dataFrame:\n")

      print(df.printSchema)
      import java.lang.Exception
      //drop null values
      val dfnNa = df.na.drop

      //filling null values to  blank
      val processedDf = dfnNa.na.fill("")
      //drop duplicates
      val processedDf1 = processedDf.dropDuplicates()

      import org.apache.spark.sql.functions._

      val toInt = udf[Int, String](_.toInt)
      val toDouble = udf[Double, String](_.toDouble)

      //Data Cleansing
      val df1 = processedDf1.withColumn("TotalCharges2", toDouble(col("TotalCharges")))
      val df2 = df1.withColumn("gender2", when(col("gender") === "Female", 0).otherwise(1))
      val df3 = df2.withColumn("partner2", when(col("partner") === "No", 0).otherwise(1))
      val df4 = df3.withColumn("dependents2", when(col("dependents") === "No", 0).otherwise(1))
      val df5 = df4.withColumn("contract2", when(col("contract") === "One year", 1).when(col("contract") === "Two year", 2).otherwise(0))
      val df6 = df5.withColumn("churn", when(col("churn") === "No", 0).otherwise(1))
      //Dropping the insignificant fields
      val dfFinal = df6.drop("customerID", "gender", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "TotalCharges", "TotalCharges2");

      val dfFinal1 = dfFinal.withColumn("label", col("churn")).drop("churn")
      print("data after cleansing:")
      print(dfFinal1.show(10))
      print("\n\nSchema of required fields for prediction:\n")
      print(dfFinal1.printSchema)

      //Building NaiveBayes Model
      val rdd = dfFinal1.rdd

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
      print("\nAccuracy:" + accuracy * 100 + "\n")
    } catch {
      case ex: FileNotFoundException => {
        println("File not found Exception")
      } case ex: IOException => {
        println("IO Exception")
      }
    }
  }
}
