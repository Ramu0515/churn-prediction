package com.sample

import org.apache.spark.mllib.classification.{ NaiveBayes, NaiveBayesModel }
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession
object LinearRegModel {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("ChurnPrediction")
      .getOrCreate()
    import spark.implicits._
    import java.io.FileNotFoundException
    import java.io.IOException

    try {
      val df = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("/spl/project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

      val rdd1 = df.rdd

      //df.
      print("\nSchema of the initial dataFrame:\n")

      print(df.printSchema())
      import java.lang.Exception
      var processedDf1 = df
      try {
        //drop null values
        val dfnNa = df.na.drop

        //filling null values to  blank
        val processedDf = dfnNa.na.fill("")
        //drop duplicates
        processedDf1 = processedDf.dropDuplicates()
      } catch {
        case ex: Exception => {
          println("Exception ocuured  during pre processing")
        }
      }
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

      print("\n\nSchema of required fields for prediction:\n")
      print(dfFinal1.printSchema)

      //Building the prediction Model
      import org.apache.spark.ml.feature.{ VectorAssembler, VectorIndexer }

      val assembler = new VectorAssembler().setInputCols(Array("SeniorCitizen", "MonthlyCharges", "gender2", "partner2", "dependents2", "contract2")).setOutputCol("features")

      val dfFinalX = assembler.transform(dfFinal1)

      val splits = dfFinalX.randomSplit(Array(0.8, 0.2), 42)

      val training_data = splits(0)

      val test_data = splits(1)

      import org.apache.spark.ml.regression._
      val linearRegression = new LinearRegression()

      val model = linearRegression.fit(training_data)

      val prediction = model.transform(test_data)
      print("\n\nCoefficients:" + model.coefficients)
      print("\n\nIntercept:" + model.intercept)
      print("\n\nRoot Mean Squared Error:" + model.summary.rootMeanSquaredError)
      print("\n\nr2 value:" + model.summary.r2)
      print("\n\nAccuracy:" + (1 - model.summary.rootMeanSquaredError))

      //print(prediction.show())
      //Writing predictions into HDFS location
      import java.time.format.{ DateTimeFormatter, DateTimeParseException }

      import java.time.{ LocalDate, ZoneId, ZonedDateTime }

      import java.time.LocalDateTime

      val timestamp = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd_HHmmss"))

      prediction.drop("features").write.format("com.databricks.spark.csv").save("/spl/project/LinReg/predictions/" + timestamp + "/")
      print("Please refer /spl/project/LinReg/predictions/ path for predictions:")

    } catch {
      case ex: FileNotFoundException => {
        println("File not found Exception")
      } case ex: IOException => {
        println("IO Exception")
      }
    }
  }
}