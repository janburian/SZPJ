# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 21:44:43 2017

@author: ircing
"""
import math

qrel = 'cacm_devel.rel'          #### file with relevance judgments
scored = 'vzor_vystupu_vyhledavaciho_programu.txt'         #### file with retrieved documents

# qrel = 'toy.rel'
#scored = 'toy.out'

rel_docs = {}
retrieved_docs = {}

AP = {}

with open(qrel, 'r') as relevant_files:        ##  read the list of relevant files
    for line in relevant_files:
        line.strip()
        items=line.split(" ")
        
        if items[0] not in rel_docs.keys():
            rel_docs[items[0]] = []
        
        rel_docs[items[0]].append(items[2]) 
        #print line

with open(scored, 'r') as retrieved_files:
    for line in retrieved_files:
        line.strip()
        items=line.split("\t")
        
        if items[0] not in retrieved_docs.keys():
            retrieved_docs[items[0]]=[]
        
        retrieved_docs[items[0]].append(items[1])
        
        #print line
        
### compute average precisions for individual topics first
acc_MAP = 0

for topic in retrieved_docs:
    acc_AP = 0
    number_of_relevant = 0
    
    for position in range(100):
        if retrieved_docs[topic][position] in rel_docs[topic]:
            number_of_relevant += 1
            this_point_P = number_of_relevant / (position + 1)
            acc_AP += this_point_P
            # print retrieved_docs[topic][position] + " " + str(this_point_P)
                                
    if (number_of_relevant == 0):
        AP[topic] = 0
    else:
        AP[topic] = acc_AP / (len(rel_docs[topic]))
    
    acc_MAP += AP[topic]
    print_output = "AP for topic " + topic + " is " + str(AP[topic])
    print(print_output)
    
MAP = acc_MAP / len(retrieved_docs)

print_map = "**** MAP is " + str(MAP) 
print(print_map)
      