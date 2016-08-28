# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:39:15 2016

@author: priyanka
"""
#==============================================================================
# TF implementation full in spark, newjoinedRDD same as joinedRDD discussed while creating readFile script. Narita- you can directly merge the code and just float error in function div needs to be resolved..
# Unwell so idf implementation by tomm evening if condition stable.
#Output format: [docid , (total no of words in doc, dict(word: tf))]
#==============================================================================
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from collections import Counter

sc = SparkContext("local", "tfidf App")
sqlContext = SQLContext(sc)

def foo(x):
    return dict(Counter(x))
     
def div(x):
#    print "total"
#    print x[0]
#    print x[1]
    
    for k,v in x[1].items():
        #print "*********"
        #print k
        #print x[1][k]
        #print (x[1][k]/x[0])
        #float error here resolve  
        x[1][k] = float(x[1][k]/x[0])
    
    tup = (x[0],x[1])
    return tup
    
        
     
    
def main():
    
    documents = [
                    [1,'Chinese Beijing Chinese','c'],
                    [2,'Chinese Chinese Shanghai','c'],
                    [3,'Chinese Macao','c'],    
                    [4,'Tokyo Japan Chinese','j']   
                    ]
    
    #TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
    
    newjoinedRDD = sc.parallelize(documents)

    #denominator of TF calculated
    tfDenom = newjoinedRDD.map(lambda x: (x[0],len(x[1].split(" "))))

    #[(1, 3), (2, 3), (3, 2), (4, 3)]
    tfDenom.collect()

    #numerator of tf calculation
    #[(1, {'Beijing': 1, 'Chinese': 2}), (2, {'Shanghai': 1, 'Chinese': 2}), (3, {'Macao': 1, 'Chinese': 1}), (4, {'Japan': 1, 'Chinese': 1, 'Tokyo': 1})]
    tfNum= newjoinedRDD.map(lambda x:(x[0],foo(x[1].split())))

    #Join num and denominator
    tfCalc = tfDenom.join(tfNum)
    print tfCalc.collect()
   
    #[doc id,(total no of words in doc,dict(word, no of times word appears in docid))]
    #[(1, (3, {'Beijing': 1, 'Chinese': 2})), (2, (3, {'Shanghai': 1, 'Chinese': 2})), (3, (2, {'Macao': 1, 'Chinese': 1})), (4, (3, {'Japan': 1, 'Chinese': 1, 'Tokyo': 1}))]
    tfFinal = tfCalc.map(lambda x:(x[0],div(x[1])))

    print tfFinal.collect()



if __name__ == "__main__":
    main()