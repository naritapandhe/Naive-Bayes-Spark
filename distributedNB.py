from __future__ import division
import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
import re
#from stemming.porter2 import stem
#from nltk.corpus import stopwords
from collections import Counter
import math
import sys
from collections import OrderedDict



def read_stop_words():
    #readStopWords = stopwords.words("english") 
    readStopWords =  ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'yo', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn','tel','fax','would','could','should']
    #cachedStopWords = map(lambda word: word.encode("utf-8"),readStopWords)
    return readStopWords

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


def clean_word(w):
   #remove everything except aA-zZ    
   x = re.sub("'|\.|\;|\:|\?|\!","", (w.lower()))
   #return re.sub("\,|\.|\;|\:|\;|\?|\!|\[|\]|\}|\{|&quot|'|&amp|-|\d+"," ", x)
   return re.sub('\&quot|\&amp|[^a-zA-Z]'," ",x)

def clean_doc(words):
    #replace consecutive spaces with 1 space
   return re.sub("\s\s+|\+"," ", words)
   

def removeCAT(doc):
    words = doc.split(",")
    l = []
    for word in words:
        if "CAT" in word:
            l.append(word.lower())
    return l

def joinOverride(docID,docWords,broadCastedLabels):
    output = []
    labels = broadCastedLabels.get(docID)
    for label in labels:
        output.append((docID,docWords,label))
    return output   

def calPrior(x,totalNoOfDocs):
    y =float(x/totalNoOfDocs)
    return (y)    
        
def test(docID,docWords,classWiseWordCounts,wordAndDocCountPerClass,vocabSize,totalNoOfDocs):
    #final formula: math.log(prior prob of class(x)) + math.log(cp(w))
    #where (prior prob of class(x)) = (total #docs in class(x))/(total #docs)
    #cpw (count(w,c)+1)/count(c)+|vocabSize|
    docProbability = dict()
    #{CCAT,(800,21)}
    for label,value in wordAndDocCountPerClass.iteritems():
        #calculate prior
        docProbability[label] = math.log(calPrior(value[1],float(totalNoOfDocs)))
        for word in docWords:
            wordCount = classWiseWordCounts.get(label).get(word)
            if wordCount==None:
                wordCount=0
            docProbability[label] += math.log((float(wordCount+1)/(float(value[0]+vocabSize))))
    
    maxDocProb=max(docProbability, key=docProbability.get) 
    return (docID,docProbability,maxDocProb)                  



def main():

    sc = SparkContext(conf = SparkConf().setAppName("Distributed NB"))
   
    stopWords = sc.broadcast(read_stop_words())
#==============================================================================
# Read the training documents
#==============================================================================

    docData = sc.textFile('dtap://TenantStorage/students/csci8360/p1/X_train_large.txt',25).map(lambda doc:doc.encode("utf-8").strip())
    #docData = sc.textFile('X_train_small.txt',35).map(lambda doc:doc.encode("utf-8").strip())

    #create an RDD of the training documents
    #entireDocData = sc.parallelize(docData).zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()
    entireDocData = docData.zipWithIndex().map(lambda doc:(doc[1],clean_word(doc[0]))).cache()
    cleanedDocData = entireDocData.map(lambda doc:(doc[0],clean_doc(doc[1])))

#==============================================================================
# Reading the training labels
#==============================================================================
    #read the documents from training file
    labelData = sc.textFile('dtap://TenantStorage/students/csci8360/p1/y_train_large.txt',25).map(lambda doc:doc.encode("utf-8").strip())
    #labelData = sc.textFile('y_train_large.txt',35).map(lambda doc:doc.encode("utf-8").strip())
 
    #create an RDD of the labels
    #filter words suffixed with 'CAT'
    labelData1 = labelData.map(lambda doc: removeCAT(doc)).zipWithIndex().map(lambda doc:(doc[1],doc[0])).toLocalIterator()
    labelDataOrderedDict = OrderedDict((k, v) for (k,v) in labelData1)
    labelDataOrderedDictBroadCast = sc.broadcast(labelDataOrderedDict)

#==============================================================================
# Map Side Join
#==============================================================================
    mapSideJoinedRDD = cleanedDocData.flatMap(lambda doc:joinOverride(doc[0],doc[1],labelDataOrderedDictBroadCast.value))
    
    #remove stop words from the training documents
    #labelsAndDocJoinedRDD = mapSideJoinedRDD.map(lambda doc:(doc[0], ([word for word in doc[1].split() if word not in stopWords.value]),doc[2])).cache()
    labelsAndDocJoinedRDD = mapSideJoinedRDD.map(lambda doc:(doc[2], ([word for word in doc[1].split() if word not in stopWords.value]))).cache()
    labelsAndDocJoinedRDD.take(1)
    
#==============================================================================
# Part 3: Vocab Size; |V|
#==============================================================================
    totalNoOfDocs = labelsAndDocJoinedRDD.count()
    totalNoOfDocsBroadCast = sc.broadcast(totalNoOfDocs)

    #Find the vocab
    vocabSize = labelsAndDocJoinedRDD.flatMap(lambda doc:doc[1]).distinct().count()
    vocabSizeBroadCast = sc.broadcast(vocabSize)
 
#==============================================================================
# Part 2: Counts for Priors; count(c)
#==============================================================================
    #Returns: {CCAT,(800,21)}
    wordAndDocCountPerClass = labelsAndDocJoinedRDD.combineByKey(lambda wordList: (len(wordList), 1),
                                lambda wordAndDocCount, wordList: (wordAndDocCount[0] + len(wordList), wordAndDocCount[1] + 1),
                                lambda wordCount, docCount: (wordCount[0] + docCount[0], wordCount[1] + docCount[1])).collectAsMap()

    wordAndDocCountPerClassBroadCast = sc.broadcast(dict(wordAndDocCountPerClass))
    #print "wordAndDocCountPerClassBroadCast!!!!!"
    #print wordAndDocCountPerClassBroadCast.value
    print "wordAndDocCountPerClassBroadCast!!!!!"
#==============================================================================
# Part 1: Counts for Conditional Probs; count(w,c)
# Formula: (count(w,c)+1)/count(c)+|V|
# Below returns: (CCAT,({w1:c1,w2:c2},21))
#==============================================================================
    individualWordCountPerClass = labelsAndDocJoinedRDD.combineByKey(lambda wordList: (Counter(wordList), 1),
                                lambda individualWordCount, wordList: (individualWordCount[0] + Counter(wordList), individualWordCount[1] + 1),
                                lambda wordCount, docCount: (wordCount[0] + docCount[0], wordCount[1] + docCount[1]))

    individualWordCountPerClassDict = individualWordCountPerClass.map(lambda (x,y):(x,dict(y[0]))).toLocalIterator()
    individualWordCountPerClassDictBroadCast = sc.broadcast(dict(individualWordCountPerClassDict))
    #print "individualWordCountPerClassDictBroadCast!!!!!"
    #print individualWordCountPerClassDictBroadCast.value
    print "individualWordCountPerClassDictBroadCast!!!!!"
#==============================================================================
# Testing
#==============================================================================       
    print "testing begins!!!!!"
    #read the documents from training file
    testDocData =  sc.textFile('dtap://TenantStorage/studets/csci8360/p1/X_test_large.txt',25).map(lambda doc:doc.encode("utf-8").strip())
    #testDocData =  sc.textFile('X_test_large.txt',35).map(lambda doc:doc.encode("utf-8").strip())
    
    #create an RDD of the training documents
    #testEntireDocData = sc.parallelize(testDocData).zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()
    testEntireDocData = testDocData.zipWithIndex().map(lambda doc:(doc[1],clean_word(doc[0]))).cache()
    cleanedTestDocData = testEntireDocData.map(lambda doc:(doc[0],clean_doc(doc[1])))
    cleanedTestDocData.take(1)
    
    #remove stop words from the training documents: returns (0,['w1','w2','w3'])
    stopwordsRemovedTestDocData = cleanedTestDocData.map(lambda doc:(doc[0], ([word for word in doc[1].split() if word not in stopWords.value])))
    
    wordsSplittedOfTestDocData = stopwordsRemovedTestDocData.map(lambda doc:test(doc[0],doc[1],individualWordCountPerClassDictBroadCast.value,wordAndDocCountPerClassBroadCast.value,vocabSizeBroadCast.value,totalNoOfDocsBroadCast.value))
    wordsSplittedOfTestDocData.take(1)

    results = wordsSplittedOfTestDocData.map(lambda doc:doc[2])
    print "results found!!!!!"
    x = results.toLocalIterator()
       
    orig_stdout = sys.stdout
    f = file('naiveBayesOutput.txt', 'w')
    sys.stdout = f

    for i in x:
        print i.upper()

    sys.stdout = orig_stdout
    f.close() 
   

if __name__ == "__main__":
    main()


