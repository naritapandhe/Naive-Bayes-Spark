# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 18:02:41 2016

@author: priyanka
"""

#cleanLabel play v1

#this script takes in y(label) file as input, extracts the labels with CAT and outputs it in a text file.
#will be ported to spark later

j = 0
#Input file
fname = "/Users/priyanka/Desktop/y_train_vsmall.txt"

#Output file
outfile = open('/Users/priyanka/Desktop/y_train_vsmall_out.txt','w')

for l in open(fname):
    #print j
    s = l.rstrip('\n')
    s1 = s.split(",")
    #extract words with CAT from the list
    s2 = [x for x in s1 if "CAT" in x]
    #print s2
    j = j+1
    if not s2:
        print "empty list"
        outfile.write("empty list")
    else:
        print s2
        i = 0
        for item in s2:
            i = i + 1
            #print i
            if i != len(s2):
                item = item + ","
            
            outfile.write(item)
        
    outfile.write("\n")
            
outfile.close()

#j - number of lines in file
print j 
        
