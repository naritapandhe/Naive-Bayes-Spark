# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 21:32:07 2016

@author: priyanka
"""

#==============================================================================
# This script takes in the label file from url and creates RDD with <index,label>. It involves:
#1. Read label file from URL
#2. Creating RDD
#3. Removing labels without "CAT"
#4. Adding index
#5. Duplicating <index,label> with multiple labels.
#==============================================================================
import urllib
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row



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

def removeCAT(doc):
    l = []
    for item in doc:
        if "CAT" in item:
            l.append(item)
        
    return l
    

def deduplicate(d1,d2):
    l = []
    for item in d2:
            tup = (d1,item)
            l.append(tup)
    return l

    
def main():
        #read the documents from training file
     lData = read_document_data('https://s3.amazonaws.com/eds-uga-csci8360/data/project1/y_train_vsmall.txt')
 
     
     entireLabelData = sc.parallelize(lData).cache()
     
     removeEndline = entireLabelData.map(lambda doc: doc.rstrip('\n'))
     
     
     #convert labels into list
     listData= removeEndline.map(lambda doc: doc.split(","))
    
     #remove CCAT
     labelData = listData.map(lambda doc: removeCAT(doc))
     
     #add index,swap and cache
     indexData = labelData.zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()
     
     #add index with each corresponding label separately 
     dData1 = indexData.map(lambda doc:deduplicate(doc[0],doc[1]))
     
     #not printing tuple no for empty tuple.
     print dData1.collect()
     
if __name__ == "__main__":
    main()
    
     