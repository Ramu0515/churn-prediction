#get the location/column_number of target
def get_loc(list_of_columns,column_name):
	i=0
	for each_col in list_of_columns:
		if(each_col==column_name):
			return i+1
		i=i+1	

#calculate the theta values
def weights_calculation(line):
	z=np.asmatrix(line[:targetcolno-1])
	k=np.asmatrix(line[targetcolno:len(line)])
	X_matrix=np.concatenate((np.asmatrix([1.]),z,k),axis=1)
	Y_matrix=line[targetcolno-1]
	dot_product=np.dot(X_matrix,initial_weights)
	sigmoid=1/(1+np.exp(-dot_product))
	y_pred=sigmoid.item(0)
	error=sigmoid-Y_matrix
	for element in range(0,X_matrix.shape[1]):
		each_gradient=X_matrix[0,element]
		value=((Y_matrix*(1-y_pred)*each_gradient)-((1-Y_matrix)*y_pred*each_gradient))
		yield( str(element),(value))
		
		
import sys
if(len(sys.argv)!=4):
	print("Usage: spark-submit --packages com.databricks:spark-csv_2.10:1.5.0 LogisticRegression1.py inputFile iterations stepval")
	sys.exit(-1)


from pyspark import SparkContext
sc=SparkContext()
import numpy as np
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

#read the input file
df = sqlContext.read.format("csv").option("header","true").option("inferSchema","true").load(sys.argv[1])



df1=df.select(['rev_Mean','mou_Mean','totmrc_Mean','churn'])

df1.cache()



(trainingData, test) = df1.randomSplit([0.6, 0.4])

df12=trainingData
df12=df1.na.fill(0)

targetcolno=get_loc(df1.columns,"churn")
initial_weights=np.zeros(len(df12.columns))


from operator import add

for iterations in range(0,int(sys.argv[2])):
	lr=float(sys.argv[3])
	gradientweights=df12.flatMap(lambda each_line:weights_calculation(each_line)).cache()
	final_gradients=gradientweights.reduceByKey(add)
	k=final_gradients.sortByKey(ascending=True).values().collect()
	
	new_weights=np.zeros(len(df1.columns))
	for i in range(0,len(new_weights)):
		new_weights[i]=(initial_weights[i]-k[i]*lr)
		
	initial_weights=new_weights	
	print("****-----------------**********")
	print("weights/coeffecients at iteration "+str(iterations)+":"+str(initial_weights))
	print("****-----------------**********")

#predict on test data	
def predict(line):
	z=np.asmatrix(line[:targetcolno-1])
	k=np.asmatrix(line[targetcolno:len(line)])
	X_matrix=np.concatenate((np.asmatrix([1.]),z,k),axis=1)
	Y_matrix=line[targetcolno-1]
	dot_product=np.dot(X_matrix,initial_weights)
	sigmoid=1/(1+np.exp(-dot_product))
	yield Y_matrix,sigmoid.item(0)

	
test=test.na.fill(0)	
output=test.flatMap(lambda eachLine:predict(eachLine))	
correct=output.filter(lambda (x, v): x == v).count()
accuracy=(1.0*correct)/output.count()
print("Accuracy on the final dataset--->"+str(accuracy))


#clean the memory
df.unpersist()
df12.unpersist()
trainingData.unpersist()
test.unpersist()


sc.stop()