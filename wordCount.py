import urllib
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from collections import Counter
import re
from collections import OrderedDict
import sys


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

def read_stop_words():
    with open("C:/DSP_Project1/alias-project1/stopWordList.txt") as f:
        readStopWords = [x.strip('\n').encode("utf-8") for x in f.readlines()]

    return readStopWords 

def clean_word(w):
   #remove everything except aA-zZ    
   x = re.sub("'|\.|\;|\:|\?|\!|\-","",(w.lower()))
   #return re.sub("\,|\.|\;|\:|\;|\?|\!|\[|\]|\}|\{|&quot|'|&amp|-|\d+"," ", x)
   return re.sub('\&quot|\&amp|[^a-zA-Z]'," ",x)

def clean_doc(words):
	#replace consecutive spaces with 1 space
   return re.sub("\s\s+|\+"," ", words)

def main():
	#read the labels from,words testing file
	docData = read_document_data('https://s3.amazonaws.com/eds-uga-csci8360/data/project1/X_train_vsmall.txt')
	entireDocData = list(map(lambda doc:(clean_word(doc)),docData))
	cleanedDocData= list(map(lambda doc1:(clean_doc(doc1)),entireDocData))
	entireCleanedData= "".join(cleanedDocData)
	listData= entireCleanedData.split()
	stopWords = read_stop_words()
	#print stopWords
	dataClean= list(map(lambda doc2:([if doc2 not in stopWords]),listData))
	print dataClean
	
	# res = OrderedDict(Counter(listData))
	# orig_stdout= sys.stdout
	# f = file('countsOutput.txt', 'w')
	# sys.stdout = f

	# for key, value in sorted(res.items(), key=lambda k: k[1], reverse=True):
	# 	print (key,value)

	# sys.stdout = orig_stdout
	# f.close() 
	#print res
	

if __name__ == "__main__":
	main()