# churn-prediction
Cloud computing

Execute  the spark job by below commands
1. for Logistic regression:
spark-submit --packages com.databricks:spark-csv_2.10:1.5.0 LogisticRegression1.py inputFile iterations stepval
2. for Naive Bayes classification:
 spark-submit --packages com.databricks:spark-csv_2.10:1.3.0 naivebayes.py $filename
