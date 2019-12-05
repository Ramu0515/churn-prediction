from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
df = sqlContext.read.format("csv").option("header","true").option("inferSchema","true").load("Telecom_customerchurn.csv")

drops=["dwllsize","forgntvl","ethnic","kid0_2","kid3_5","kid6_10","kid11_15","kid16_17","creditcd","Customer_ID"]

stringcols = [item[0] for item in df.dtypes if  item[1].startswith('string')]
stringcols.append("Customer_ID");
for col in stringcols:
	print(col)
	if (col not in drops):
		indexer = StringIndexer(inputCol=col, outputCol=col+"_numeric").fit(df)
		df = indexer.transform(df)
	df=df.drop(col)
	print("finished")
	
df=df.na.drop(subset=df.columns)

df=df.na.fill(0)

rdd = df.rdd.map(list)


from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel


rdd3=rdd.map(lambda line:LabeledPoint(float(line[48]),Vectors.dense(float(line[1]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7]),float(line[8]),float(line[11]),float(line[12]),float(line[13]),float(line[14]),float(line[15]),float(line[16]),float(line[17]),float(line[18]),float(line[19]),float(line[20]),float(line[21]),float(line[22]),float(line[23]),float(line[24]),float(line[25]),float(line[26]),float(line[27]),float(line[28]),float(line[29]),float(line[30]),float(line[31]),float(line[32]),float(line[33]),float(line[34]),float(line[35]),float(line[36]),float(line[37]),float(line[38]),float(line[39]),float(line[40]),float(line[41]),float(line[42]),float(line[43]),float(line[44]),float(line[45]),float(line[46]),float(line[47]),float(line[49]),float(line[50]),float(line[51]),float(line[52]),float(line[53]) ,float(line[54]) ,float(line[55]) ,float(line[56]) ,float(line[57]) ,float(line[58]) ,float(line[59]) ,float(line[60]) ,float(line[61]) ,float(line[62]) ,float(line[63]) ,float(line[64]) ,float(line[65]) ,float(line[66]) ,float(line[67]) ,float(line[68]) ,float(line[69]) ,float(line[70]) ,float(line[71]) ,float(line[72]) ,float(line[73]) ,float(line[74]) ,float(line[75]) ,float(line[76]) ,float(line[77]) ,float(line[78]) ,float(line[79]) ,float(line[80]) ,float(line[81]) ,float(line[82]) ,float(line[83]) ,float(line[84]) ,float(line[85]) ,float(line[86]) ,float(line[87]) ,float(line[88]) ,float(line[89]) ,float(line[90]))))

training, test = rdd3.randomSplit([0.6, 0.4], seed = 0)

model = NaiveBayes.train(training, 1.0)


predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))

accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count()/test.count()

print(accuracy)
