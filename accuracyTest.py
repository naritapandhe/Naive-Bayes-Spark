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
    returnResults = []
    for word in labelList:
        if output == word.lower():
            returnResults.append(('T',1))
            break
        
    if len(returnResults) == 0:
        returnResults.append(('F',1))

    return returnResults   
    
    
def main():
     #read the labels from testing file
     testData = read_document_data('https://s3.amazonaws.com/eds-uga-csci8360/data/project1/y_test_small.txt')
     
     entireLabelData = sc.parallelize(testData).zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()
     removeEndline = entireLabelData.map(lambda doc: (doc[0],doc[1].rstrip('\n')))

     
     #read the labels from outut file
     olist = sc.textFile("/Users/admin/Documents/pythonworkspace/data-science-practicum/project-1/alias-project1/oFile.txt")
     olist1 = olist.map(lambda word:word.encode("utf-8"))
     
     entireOtputData= olist1.zipWithIndex().map(lambda testDoc:(testDoc[1],testDoc[0])).cache()

     joinedRDD = removeEndline.join(entireOtputData)
     labelsAndOutputJoinedRDD = joinedRDD.map(lambda (x,y):(x,y[0],y[1]))


     resulatantData= labelsAndOutputJoinedRDD.map(lambda doc:checkIfExist(doc[1].split(','),doc[2])).flatMap(lambda x:x).reduceByKey(lambda x,y:x+y)
     print resulatantData.collect()
     

if __name__ == "__main__":
    main()