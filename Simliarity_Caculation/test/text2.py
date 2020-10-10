#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

a = ["cat","eat","mat","look","cat"]
b = ["cat","sit","seat","look"]

lena = len(a)
lenb = len(b)

c = np.ones((lena,lenb),dtype = float) 
print(c)

for i in range(lena):
    for j in range(lenb):
        c[i,j] = np.random.random() 
print(c)


a = [1,3,2,2]

print(list(set(a)))

if 1 in a:
    print("1 is in the list")



n = 0
wei = 0
for word2 in b:
    for word1 in a:
       if word1 == word2:
           n+=1
           wei += 1
           break
print(wei,n)
'''
def fetchmax(c,lena,lenb,sim_sum):
    max_s = 0 
    pos_line = 0
    pos_col = 0
    for i in range(lena):
        for j in range(lenb):
            if c[i,j] >= max_s:
                max_s = c[i,j]
                pos_line = i
                pos_col = j
    sim_sum += max_s
    lena-=1
    lenb-=1
    c = np.delete(c,pos_line,axis = 0)
    if lena == 0:
        return c,lena,lenb,sim_sum
    c = np.delete(c,pos_col,axis = 1)
    return c,lena,lenb,sim_sum
    
sim_sum = 0.
iter_time = 0
while lena > 0 and lenb > 0:
    c,lena,lenb,sim_sum = fetchmax(c,lena,lenb,sim_sum)
    print(c) 
    iter_time += 1

print("the ave result is ",sim_sum/iter_time)
'''




