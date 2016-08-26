# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:39:15 2016

@author: priyanka
"""
#==============================================================================
# TF -IDF implementation full in spark, 
# newjoinedRDD same as joinedRDD discussed while creating readFile script. 
# Narita- you can directly merge the code 
# just float error in function div needs to be resolved.
# idfNum calculation can be optimised 
# tF idf Output format: [('Japan', 4, <tf idf value>),....]
# Output format: [(word,doc id, tf-idf val)]
# values in final output will show 0 as we have to fix function div float.
# TF Output format : [(word,(docid,tf))]
# IDF Output format: [(word,idf)]
#==============================================================================
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from collections import Counter
import math

sc = SparkContext("local", "tfidf App")
sqlContext = SQLContext(sc)


def foo(x):
    return dict(Counter(x))
    
def foo1(x):
    Counter({'x':1})
    return dict(Counter(x))
    
def foo2(x):
    #print x
    l = []
    for k,v in x.items():
        x[k] = 1
        l.append(k)
    return tuple(l)
    
     
def div(x):
    l = []
    for k,v in x[1].items():
        #float error here resolve  
        x[1][k] = float(x[1][k]/x[0])
        l.append((k,x[1][k]))
    
    #print "********"
    #print l
   # tup = (x[0],x[1])
    return tuple(l)
    
        
     
    
def main():
    
    documents = [
                    [1,'Chinese Beijing Chinese','c'],
                    [2,'Chinese Chinese Shanghai','c'],
                    [3,'Chinese Macao','c'],    
                    [4,'Tokyo Japan Chinese','j']   
                    ]
#==============================================================================
# TF Calculation
# TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
#==============================================================================
    
    
    newjoinedRDD = sc.parallelize(documents)

    #denominator of TF calculated
    tfDenom = newjoinedRDD.map(lambda x: (x[0],len(x[1].split(" "))))

    #[(1, 3), (2, 3), (3, 2), (4, 3)]
    tfDenom.collect()

    #numerator of tf calculation
    #[(1, {'Beijing': 1, 'Chinese': 2}), (2, {'Shanghai': 1, 'Chinese': 2}), (3, {'Macao': 1, 'Chinese': 1}), (4, {'Japan': 1, 'Chinese': 1, 'Tokyo': 1})]
    tfNum= newjoinedRDD.map(lambda x:(x[0],foo(x[1].split())))

    #Join num and denominator
    #[(1, (3, {'Beijing': 1, 'Chinese': 2})), (2, (3, {'Shanghai': 1, 'Chinese': 2})), (3, (2, {'Macao': 1, 'Chinese': 1})), (4, (3, {'Japan': 1, 'Chinese': 1, 'Tokyo': 1}))]
    tfCalc = tfDenom.join(tfNum)
    #print tfCalc.collect()
   
    # Input - [(doc id,(total no of words in doc,dict(word, no of times word appears in docid)))]
    # Output - [(1, (('Beijing', 0.0), ('Chinese', 0.0))), (2, (('Shanghai', 0.0), ('Chinese', 0.0))), (3, (('Macao', 0.0), ('Chinese', 0.0))), (4, (('Japan', 0.0), ('Chinese', 0.0), ('Tokyo', 0.0)))]
    tfFinal = tfCalc.map(lambda x:(x[0],div(x[1])))
    
    #[(word,(docid,tf))]
    #[('Beijing', (1, 0.0)), ('Chinese', (1, 0.0)), ('Shanghai', (2, 0.0)), ('Chinese', (2, 0.0)), ('Macao', (3, 0.0)), ('Chinese', (3, 0.0)), ('Japan', (4, 0.0)), ('Chinese', (4, 0.0)), ('Tokyo', (4, 0.0))]
    tfFormat = tfFinal.flatMapValues(lambda x:x).map(lambda x:(x[1][0],(x[0],x[1][1])))
    print tfFormat.collect()


#==============================================================================
# IDF calcultion
# IDF(t) = log_e(Total number of documents / Number of documents with term t in it). 
#==============================================================================

    #4
    idfNum = newjoinedRDD.count()
    #print idfNum
    
    #this function can be optimised
    #[('Beijing', 1), ('Chinese', 4), ('Tokyo', 1), ('Shanghai', 1), ('Japan', 1), ('Macao', 1)]
    idfDenom = newjoinedRDD.map(lambda x: (x[0],foo1(x[1].split()))).map(lambda y: foo2(y[1])).flatMap(lambda word:word).map(lambda z:(z,1)).reduceByKey(lambda x,y:x+y)
    #print idfDenom.collect()
   
    #[('Beijing', 1.3862943611198906), ('Chinese', 0.0), ('Tokyo', 1.3862943611198906), ('Shanghai', 1.3862943611198906), ('Japan', 1.3862943611198906), ('Macao', 1.3862943611198906)]
    idfCalc = idfDenom.map(lambda x: (x[0],math.log(idfNum/x[1])))
    
    print idfCalc.collect()
#==============================================================================
# TF-IDF Total
#==============================================================================  
    
    #[('Japan', ((4, 0.0), 1.3862943611198906)), ('Macao', ((3, 0.0), 1.3862943611198906)), ('Tokyo', ((4, 0.0), 1.3862943611198906)), ('Beijing', ((1, 0.0), 1.3862943611198906)), ('Shanghai', ((2, 0.0), 1.3862943611198906)), ('Chinese', ((1, 0.0), 0.0)), ('Chinese', ((2, 0.0), 0.0)), ('Chinese', ((3, 0.0), 0.0)), ('Chinese', ((4, 0.0), 0.0))]
    joinedtfIdf = tfFormat.join(idfCalc)
    joinedtfIdf.collect()
    
    #ctfidf = joinedtfIdf.map(lambda x: (x[0],x[1][0][0],x[1][0][1],x[1][1]))
    #ctfidf.collect()
    #[('Japan', 4, 0.0), ('Macao', 3, 0.0), ('Tokyo', 4, 0.0), ('Beijing', 1, 0.0), ('Shanghai', 2, 0.0), ('Chinese', 1, 0.0), ('Chinese', 2, 0.0), ('Chinese', 3, 0.0), ('Chinese', 4, 0.0)]
    calcTfIdf = joinedtfIdf.map(lambda x: (x[0],x[1][0][0],(x[1][0][1]*x[1][1])))
    
    print calcTfIdf.collect()


   
   
if __name__ == "__main__":
    main()