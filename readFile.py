import urllib
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
import re
from stemming.porter2 import stem
from nltk.corpus import stopwords

sc = SparkContext("local", "IndexedRDD App")
sqlContext = SQLContext(sc)


def read_document_data(url):
	#read the entire file
	txt = urllib.urlopen(url)

	#create an array of documents
	data = []
	for line in txt:
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

def main():
	#read the documents from training file
	docData = read_document_data('https://s3.amazonaws.com/eds-uga-csci8360/data/project1/X_train_vsmall.txt')
	
	#create an RDD of the training documents
	entireDocData = sc.parallelize(docData).zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()
	cleanedDocData = entireDocData.map(lambda doc:(doc[0]," ".join(map(lambda y: clean_word(y), doc[1].split())))).map(lambda doc:(doc[0],clean_doc(doc[1]).strip()))
	#print cleanedDocData.collect()

	#remove stop words from the training documents
	cachedStopWords = stopwords.words("english") 
	stopwordsRemovedDocData = cleanedDocData.map(lambda doc:(doc[0], ' '.join([word for word in doc[1].split() if word not in cachedStopWords])))
	print stopwordsRemovedDocData.take(1)


if __name__ == "__main__":
    main()

