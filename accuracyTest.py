-*- coding: utf-8 -*-

import urllib
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from collections import Counter



sc = SparkContext("local", "cleanLabel App")
sqlContext = SQLContext(sc)


def read_document_data(url):
    #read the entire file
    txt = urllib.urlopen(url)

    #create an array of documents
    data = []
    for line in txt:
        data.append(line)
    return data    

def checkIfExist(labelList,output):
     	for word in labelList:
	    if output in word
		return T
	    else
		return F
    
def main():
     #read the labels from testing file
     testData = read_document_data('https://s3.amazonaws.com/eds-uga-csci8360/data/project1/y_test_vsmall.txt')
     
     entireLabelData = sc.parallelize(testData).zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()
     removeEndline = entireLabelData.map(lambda doc: doc.rstrip('\n'))

     print removeEndline.collect()
	
     #read the labels from outut file
     list =[CCAT,GCAT,ECAT,MCAT]
     entireOtputData= sc.parallelize(list).zipWithIndex().map(lambda testDoc:(testDoc[1],testDoc[0])).cache()

     joinedRDD = removeEndline.join(entireOtputData)
     labelsAndOutputJoinedRDD = joinedRDD.map(lambda (x,y):(x,y[0],y[1]))

     resulatantData= labelsAndOutputJoinedRDD.map(lambda doc:checkIfExist(doc[1],doc[2]))

     counter= collections.Counter(resulatantData)
     print counter

if __name__ == "__main__":
    main()