# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 18:15:39 2016

@author: priyanka
"""

#==============================================================================
# This script takes output from python script cleanLabelv1.py and duplicates doc index with multiple "CAT" labels.
# Output of the form : 1 CCAT 
#                      1 MCAT 
#==============================================================================
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
sc = SparkContext("local", "Deduplication App")
sqlContext = SQLContext(sc)


def deduplicate(d1,d2):
    l = []
    for item in d2:
        #returns None if topic not present
        if "empty" in item:
            tup = (d1,[])
            l.append(tup)
            
        else:
            tup = (d1,item)
            l.append(tup)
    return l
        
def main():
    labelData = sc.textFile('/Users/priyanka/Desktop/y_train_vsmall_out.txt')
    labelData.collect()
   
    #add index,swap and cache
    indexData = labelData.zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()
    
    #convert labels to list
    dupData = indexData.map(lambda doc:(doc[0],doc[1].split(",")))
    
    #add index with each corresponding label separately 
    dData1 = dupData.map(lambda doc:deduplicate(doc[0],doc[1]))
    print dData1.collect()
    
    
if __name__ == "__main__":
    main()