##########################################################
# Scalable Document Classification using Naive Bayes on Spark
#########################################################
This was one of the projects for CSCI 8360 Data Science practicum course. We are using the Reuters news articles corpus, which is a set of news articles split into several categories. There are multiple class labels per document, but we consider just the following class labels:
* CCAT: Corporate/Industrial
* ECAT: Economics
* GCAT: Government/Social
* MCAT: Markets
 
Goal was to build Naive Bayes classifier without using any existing libraries/packages like MLLib, ML or Scikit-Learn. 

# Details
## Preprocessing
The raw input was processed in the following steps:

* Exclude all the special characters and numbers. Only alphabets were retained.
* Lowercase all the words
* Remove stopwords. Our initial list of stopwords was a classical list of stopwords. But, based on the corpus, we tried to enrich our list and build context specific list of stopwords by applying the Zipf's Law.
* We utilized unigrams and took their counts, conditioned on class, into consideration, which formed the word vectors.

## Model
We implemented the standard Multinomial Naive Bayes Classifier with Laplace(add 1) smooothing and log probabilities. This model achieved an accuracy of 94.88%

# How to run
To execute the program, run following command: 
```
path_to_spark_bin/spark-submit --master <master-url> distributedNB.py
```
The output of the program can be viewed at the same location where the program is executing. Name of the output file: naiveBayesOutput.txt

Currently, the URLs of training and testing data are hardcoded. To train and test the program on files of your choice, please change the values of: docData - Training Documents, labelData - Labels of the training documents, testDocData - Testing documents
 
# Team Members:
[Narita Pandhe](https://github.com/naritapandhe/)

Shubhi Jain

Priyanka Luthra


