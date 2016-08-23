from __future__ import division
import urllib
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
import re
#from stemming.porter2 import stem
from nltk.corpus import stopwords
from collections import Counter
import math


sc = SparkContext("local", "IndexedRDD App")
sqlContext = SQLContext(sc)


def read_data_from_url(url):
    return urllib.urlopen(url)

def read_document_data(url):
    #read the entire file
    documentText = read_data_from_url(url)

    #create an array of documents
    data = []
    for line in documentText:
        data.append(line)
    return data 

def read_document_label_data(url):
    #read the entire file
    documentLabels = read_data_from_url(url)

    #create an array of labels/document
    data = []
    for line in documentLabels:
        data.append(line)
    return data 


def clean_word(w):
   #remove everything except aA-zZ    
   x = re.sub("'|\.|\;|\:|\?|\!","", (w.lower()))
   #return re.sub("\,|\.|\;|\:|\;|\?|\!|\[|\]|\}|\{|&quot|'|&amp|-|\d+"," ", x)
   return re.sub('\&quot|\&amp|[^a-zA-Z]'," ",x)

def clean_doc(words):
    #replace consecutive spaces with 1 space
   return re.sub("\s\s+|\+"," ", words)
   

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

def clean_labels(docID,docWords):
    y = re.sub("\n","", docWords)
    x = y.split(",")
    returnResults = []
    allowedCategories = ['CCAT','ECAT','GCAT','MCAT']
    for word in x:
        if word in allowedCategories:
            returnResults.append((docID,word.lower(),docWords))
    #allowedCats = [word for word in x if word in allowedCategories]
    return returnResults

def someX(words,className):
    someList = []
    for word in words:
        x = (word,[className])
        someList.append(x)

    z=list(someList)
    return z        

def someXX(word,classDict,priorProbabilitiesBroadCast):
    for k in priorProbabilitiesBroadCast.value:
        if k not in classDict:
            classDict[k]=0
    
    return (word,classDict)     

def conditionalProb(word,wordCatCounts,priorProbabilitiesBroadCast,sizeOfVocab):
    cpForLabels = dict()
    for key,value in wordCatCounts.iteritems():
        totalNoOfWordsForKeyAsClass = priorProbabilitiesBroadCast.get(key)[0]
        cp = ((value+1)/(totalNoOfWordsForKeyAsClass+sizeOfVocab))
        cpForLabels[key] = cp
    
    return (word,cpForLabels)

def test(docID,docWords,conditionalProbsForVocabBroadCast,priorProbabilitiesBroadCast,vocabSize):
    score = dict()
    for k,v in priorProbabilitiesBroadCast.iteritems():
        score[k] = math.log(v[2])
        for word in docWords:
            if word in conditionalProbsForVocabBroadCast:
                score[k] += math.log(conditionalProbsForVocabBroadCast[word][k])
            else:
                score[k] += math.log (1/v[0]+vocabSize)

    z = max(score, key=score.get)           
    return (docID,score,z)


def main():

#==============================================================================
# Reading the training documents
#==============================================================================
    
    docData = read_document_data('https://s3.amazonaws.com/eds-uga-csci8360/data/project1/X_train_vsmall.txt')
    #create an RDD of the training documents
    entireDocData = sc.parallelize(docData).zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()
    cleanedDocData = entireDocData.map(lambda doc:(doc[0]," ".join(map(lambda y: clean_word(y), doc[1].split())))).map(lambda doc:(doc[0],clean_doc(doc[1]).strip()))
    #print cleanedDocData.collect()
    #remove stop words from the training documents
    cachedStopWords1 = stopwords.words("english") 
    myStopWords = ['the', 'that', 'to', 'as', 'there', 'has', 'and', 'or', 'is', 'not', 'a', 'of', 'but', 'in', 'by', 'on', 'are', 'it', 'if','thats']
    cachedStopWords = []

    for word in cachedStopWords1:
        cachedStopWords.append(word.encode("utf-8"))

    cachedStopWords.extend(myStopWords) 
    cachedStopWordsBroadCast = sc.broadcast(cachedStopWords)

    stopwordsRemovedDocData = cleanedDocData.map(lambda doc:(doc[0], ' '.join([word for word in doc[1].split() if word not in cachedStopWordsBroadCast.value])))


#==============================================================================
# Reading the training labels
#==============================================================================
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

    #add index with each corresponding label separately - flatMapValues
    cleanLabeldata = indexData.map(lambda doc:deduplicate(doc[0],doc[1])).flatMap(lambda x: x)
    
#==============================================================================
# Joining the documents and labels
#==============================================================================
    
    joinedRDD = stopwordsRemovedDocData.join(cleanLabeldata)
    labelsAndDocJoinedRDD = joinedRDD.map(lambda (x,y):(x,y[0],y[1]))
    
#==============================================================================
# Train the model based on training docs
#==============================================================================

    totalNoOfDocs = labelsAndDocJoinedRDD.count()


    #1) Broadcast the no. of docs to all slaves
    totalNoOfDocsBroadcast = sc.broadcast(totalNoOfDocs)

    #This will return the no.of words/class: [{CCAT:8,MCAT:3}]
    wordsPerClass = labelsAndDocJoinedRDD.map(lambda doc:(doc[2],doc[1].split())).reduceByKey(lambda x,y: x+y).cache()
    countOfWordsPerClass = wordsPerClass.map(lambda x:(x[0],dict(Counter(x[1])))).map(lambda x:(x[0],sum(x[1].values()))).cache()

    #Find the count of documents for each class
    #returns (classname,classcount)
    countOfDocumentsPerClass = labelsAndDocJoinedRDD.map(lambda doc:(doc[2],1)).reduceByKey(lambda x,y:x+y).cache()
    
    
    #Combines both counts and produces the following result: [(catName,(countOfWordsPerCat,countOfDocumentsPerCat))]
    #[('GCAT', (3950, 21)), ('CCAT', (3998, 32)), ('MCAT', (1408, 12)), ('ECAT', (2032, 12))]
    classCounts = countOfWordsPerClass.join(countOfDocumentsPerClass).cache()
    
    #Find the priorProbabilities
    #returns (classname,(countOfWordsPerCat,countOfDocumentsPerCat,priorProbability)):(c,(3,1,3/4))
    priorProbabilities = classCounts.map(lambda (eachClassName,eachClassCounts):(eachClassName,(eachClassCounts[0],eachClassCounts[1],(eachClassCounts[1]/totalNoOfDocsBroadcast.value)))).toLocalIterator()
    
    #2) Broadcast the prior probabilities
    priorProbabilitiesBroadCast = sc.broadcast(dict(priorProbabilities))
    

    #This will return the no. of times that word has appeared in each class
    # Example: ('broad', {'CCAT': 1}), ('delaware', {'ECAT': 1})
    trainingDocsVocabCounts = labelsAndDocJoinedRDD.map(lambda x: someX(x[1].split(),x[2])).flatMap(lambda x:x).reduceByKey(lambda x,y: x+y).map(lambda x:(x[0],dict(Counter(x[1]))))
    vocabSizeBroadCast = sc.broadcast(trainingDocsVocabCounts.count())
    

    
    trainingDocsVocabCountsWrtAllClasses = trainingDocsVocabCounts.map(lambda (x,y):someXX(x,y,priorProbabilitiesBroadCast))
    

    #.toLocalIterator()
    conditionalProbsForVocab = trainingDocsVocabCountsWrtAllClasses.map(lambda (word,wordCatCounts): conditionalProb(word,wordCatCounts,priorProbabilitiesBroadCast.value,vocabSizeBroadCast.value)).toLocalIterator()
    conditionalProbsForVocabBroadCast = sc.broadcast(dict(conditionalProbsForVocab))

#==============================================================================
# All we need now is: conditionalProbsForVocabBroadCast, priorProbabilitiesBroadCast, vocabSizeBroadCast
# Test the model
#==============================================================================

    #read the documents from training file
    testDocData = read_document_data('https://s3.amazonaws.com/eds-uga-csci8360/data/project1/X_test_vsmall.txt')

    #create an RDD of the training documents
    testEntireDocData = sc.parallelize(testDocData).zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()
    cleanedTestDocData = testEntireDocData.map(lambda doc:(doc[0]," ".join(map(lambda y: clean_word(y), doc[1].split())))).map(lambda doc:(doc[0],clean_doc(doc[1]).strip()))
    
    #remove stop words from the training documents
    stopwordsRemovedTestDocData = cleanedTestDocData.map(lambda doc:(doc[0], ' '.join([word for word in doc[1].split() if word not in cachedStopWordsBroadCast.value])))
    wordsSplittedOfTestDocData = stopwordsRemovedTestDocData.map(lambda doc:test(doc[0],doc[1].split(),conditionalProbsForVocabBroadCast.value,priorProbabilitiesBroadCast.value,vocabSizeBroadCast.value))
    
    results = wordsSplittedOfTestDocData.map(lambda doc:doc[2]).collect()
    for i in results:
        print i
    
   

if __name__ == "__main__":
    main()

